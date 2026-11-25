#include "server.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <ctime>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>

namespace mllm {

using json = nlohmann::json;

// Generated text with the markers cut out: stop strings end it, and for chat "</think>" ends the reasoning (the
// template opens "<think>\n", so a reply starts inside it). A tail that could be the start of a marker waits for the
// next text. Engine thread only.
struct Splitter {
    std::vector<std::string> stops;
    bool reasoning = false;  // chat: text so far is reasoning_content
    bool content = false;    // any content emitted (its leading newlines are dropped)
    std::string held;

    struct Piece {
        std::string text;
        bool reasoning;
    };
    // the text to emit now; stopped = a stop string was found (it and the rest are dropped)
    std::vector<Piece> push(const std::string& t, bool flush, bool& stopped) {
        std::vector<Piece> out;
        held += t;
        auto put = [&](std::string s) {
            if (!reasoning && !content) s.erase(0, s.find_first_not_of('\n'));  // npos clears
            if (s.empty()) return;
            content |= !reasoning;
            out.push_back({std::move(s), reasoning});
        };
        for (;;) {
            size_t at = std::string::npos, len = 0;
            bool think = false;
            for (const std::string& m : stops)
                if (size_t i = held.find(m); i < at) at = i, len = m.size();
            if (reasoning)
                if (size_t i = held.find("</think>"); i < at) at = i, len = 8, think = true;
            if (at == std::string::npos) break;
            put(held.substr(0, at));
            held.erase(0, at + len);
            if (!think) return stopped = true, held.clear(), out;
            reasoning = false;
        }
        size_t keep = 0;  // the longest tail that is a proper prefix of a marker
        if (!flush) {
            auto tail = [&](const std::string& m) {
                for (size_t k = std::min(m.size() - 1, held.size()); k > keep; k--)
                    if (held.compare(held.size() - k, k, m, 0, k) == 0) return (void)(keep = k);
            };
            for (const std::string& m : stops) tail(m);
            if (reasoning) tail("</think>");
        }
        put(held.substr(0, held.size() - keep));
        held.erase(0, held.size() - keep);
        return out;
    }
};

struct Server::Stream {
    bool chat = false;
    std::string prompt;
    std::vector<Message> messages;
    GenParams p;
    bool default_max = false;  // chat without max_tokens: up to 4096, within max_ctx
    Splitter split;  // engine thread only
    int id = -1;     // engine request id, set before started

    std::mutex m;  // guards the rest
    std::condition_variable cv;
    bool started = false;  // submitted (prompt_tokens set) or rejected (error set)
    int prompt_tokens = 0, completion_tokens = 0, cached = 0;
    std::string error;
    std::deque<Splitter::Piece> q;
    std::string finish;  // set last: stop | length | error

    void put(auto&& f) {
        {
            std::lock_guard l(m);
            f();
        }
        cv.notify_one();
    }
};

static std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

// prompt bytes: the engine thread tokenizes (~170 ms per MB), stalling every stream meanwhile; max_ctx 16384 tokens
// of text is well under this
constexpr size_t MAX_BODY = 1 << 20;

static void fail(httplib::Response& res, int status, const std::string& msg) {
    res.status = status;
    res.set_content(dump({{"error", {{"message", msg}, {"type", "invalid_request_error"}}}}), "application/json");
}

// The request's fields into s; an error message, or "" when it is fine. null = absent (as openai-python sends).
// ponytail: other fields (n, echo, logprobs, tools, ...) are ignored, not rejected.
static std::string parse(json& j, Server::Stream& s, bool& stream, bool& usage) {
    if (!j.is_object()) return "need a JSON object";
    for (auto it = j.begin(); it != j.end();) it = it->is_null() ? j.erase(it) : std::next(it);
    if (!s.chat && !(j.contains("prompt") && j["prompt"].is_string())) return "need a string prompt";
    if (s.chat) {
        if (!(j.contains("messages") && j["messages"].is_array())) return "need a messages array";
        for (const json& m : j["messages"]) {
            if (!(m.is_object() && m.contains("role") && m["role"].is_string())) return "messages: need a role";
            std::string text;  // a string, or an array of text parts
            const json c = m.value("content", json(""));
            if (c.is_string()) text = c;
            else if (c.is_array())
                for (const json& part : c) {
                    if (!(part.is_object() && part.value("type", "") == "text" && part.contains("text") &&
                          part["text"].is_string()))
                        return "messages: content parts must be text";
                    text += part["text"].get<std::string>();
                }
            else if (!c.is_null()) return "messages: content must be a string or text parts";
            s.messages.push_back({m["role"], text});
        }
    }
    const char* max_key = j.contains("max_completion_tokens") ? "max_completion_tokens" : "max_tokens";
    if (j.contains(max_key) &&
        !(j[max_key].is_number_integer() && j[max_key] >= 1 && j[max_key] <= std::numeric_limits<int>::max()))
        return std::string(max_key) + ": need an integer >= 1";
    for (const char* k : {"temperature", "top_p"})
        if (j.contains(k) && !j[k].is_number()) return std::string(k) + ": need a number";
    if (j.contains("stop")) {
        json st = j["stop"].is_string() ? json::array({j["stop"]}) : j["stop"];
        if (!st.is_array() || st.size() > 4) return "stop: need a string or up to 4 strings";
        for (const json& x : st) {
            if (!x.is_string() || x.get<std::string>().empty()) return "stop: need non-empty strings";
            s.split.stops.push_back(x);
        }
    }
    if (!s.chat) s.prompt = j["prompt"];
    s.default_max = s.chat && !j.contains(max_key);
    s.p = {(int)j.value(max_key, (int64_t)256), j.value("temperature", 0.0f), j.value("top_p", 0.0f)};
    s.split.reasoning = s.chat;
    stream = j.value("stream", false);
    const json opts = j.value("stream_options", json::object());
    usage = opts.is_object() && opts.value("include_usage", false);
    return "";
}

void Server::handle(const httplib::Request& req, httplib::Response& res, bool chat) {
    auto s = std::make_shared<Stream>();
    s->chat = chat;
    bool stream = false, usage = false;
    try {
        json j = json::parse(req.body);
        if (std::string err = parse(j, *s, stream, usage); !err.empty()) return fail(res, 400, err);
    } catch (const json::exception& x) {
        return fail(res, 400, x.what());
    }
    {
        std::lock_guard l(m);
        if (stopping) return fail(res, 503, "shutting down");
        jobs.push_back(s);
    }
    cv.notify_one();
    {
        std::unique_lock l(s->m);
        s->cv.wait(l, [&] { return s->started; });
        if (!s->error.empty()) return fail(res, 400, s->error);
    }
    auto cancel = [this, s] {
        {
            std::lock_guard l(m);
            cancels.push_back(s);
        }
        cv.notify_one();
    };
    const json head = {{"id", (chat ? "chatcmpl-" : "cmpl-") + std::to_string(s->id)},
                       {"object", chat ? (stream ? "chat.completion.chunk" : "chat.completion") : "text_completion"},
                       {"created", (long)std::time(nullptr)}, {"model", model}};
    auto usage_json = [s] {
        return json{{"prompt_tokens", s->prompt_tokens}, {"completion_tokens", s->completion_tokens},
                    {"total_tokens", s->prompt_tokens + s->completion_tokens},
                    {"prompt_tokens_details", {{"cached_tokens", s->cached}}}};
    };
    auto choice = [chat](const std::string& text, bool reasoning, const json& finish) {
        if (!chat) return json{{"index", 0}, {"text", text}, {"logprobs", nullptr}, {"finish_reason", finish}};
        json delta = json::object();
        if (!text.empty()) delta[reasoning ? "reasoning_content" : "content"] = text;
        return json{{"index", 0}, {"delta", delta}, {"logprobs", nullptr}, {"finish_reason", finish}};
    };

    if (!stream) {
        std::string text, reasoning;
        std::unique_lock l(s->m);
        // wake every 100 ms: a client that hung up is cancelled, not left generating
        while (!s->cv.wait_for(l, std::chrono::milliseconds(100), [&] { return !s->finish.empty(); }))
            if (req.is_connection_closed()) {
                l.unlock();
                return cancel();
            }
        if (s->finish == "error") return fail(res, 500, "aborted: a lane failed");
        for (const auto& [t, r] : s->q) (r ? reasoning : text) += t;
        json j = head;
        json c = {{"index", 0}, {"logprobs", nullptr}, {"finish_reason", s->finish}};
        if (chat) {
            c["message"] = {{"role", "assistant"}, {"content", text}};
            if (!reasoning.empty()) c["message"]["reasoning_content"] = reasoning;
        } else {
            c["text"] = text;
        }
        j["choices"] = {c};
        j["usage"] = usage_json();
        res.set_content(dump(j), "application/json");
        return;
    }

    auto first = std::make_shared<bool>(true);
    res.set_chunked_content_provider(
        "text/event-stream",
        [s, head, usage, chat, first, choice, usage_json](size_t, httplib::DataSink& sink) {
            std::deque<Splitter::Piece> pieces;
            std::string finish;
            {
                // wake every 100 ms with nothing to send: httplib checks the peer between calls, so a client
                // that hung up before its first token is cancelled now, not when that token comes
                std::unique_lock l(s->m);
                if (!s->cv.wait_for(l, std::chrono::milliseconds(100),
                                    [&] { return !s->q.empty() || !s->finish.empty(); }))
                    return true;
                pieces.swap(s->q);
                finish = s->finish;
            }
            auto send = [&](const json& j) {
                const std::string line = "data: " + dump(j) + "\n\n";
                return sink.write(line.data(), line.size());
            };
            if (chat && std::exchange(*first, false)) {  // OpenAI's first chunk: the role
                json j = head;
                j["choices"] = {{{"index", 0}, {"delta", {{"role", "assistant"}, {"content", ""}}},
                                 {"logprobs", nullptr}, {"finish_reason", nullptr}}};
                if (!send(j)) return false;
            }
            for (const auto& [t, r] : pieces) {
                json j = head;
                j["choices"] = {choice(t, r, nullptr)};
                if (!send(j)) return false;
            }
            if (finish.empty()) return true;
            json j = head;
            if (finish == "error") j["error"] = {{"message", "aborted: a lane failed"}};
            else j["choices"] = {choice("", false, finish)};
            if (!send(j)) return false;
            if (usage) {
                j = head;
                j["choices"] = json::array();
                j["usage"] = usage_json();
                if (!send(j)) return false;
            }
            const std::string end = "data: [DONE]\n\n";
            if (!sink.write(end.data(), end.size())) return false;
            sink.done();
            return true;
        },
        [cancel](bool success) {  // the stream ended; not cleanly = the client went away
            if (!success) cancel();
        });
}

Server::Server(Engine& e, std::string model_name)
    : e(e), model(std::move(model_name)), http(std::make_unique<httplib::Server>()) {
    // httplib's default is SO_REUSEPORT on macOS: a second server on a busy port would bind and share it
    http->set_socket_options([](socket_t sock) { httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1); });
    http->set_payload_max_length(MAX_BODY);
    http->set_error_handler([](const httplib::Request&, httplib::Response& res) {  // httplib's own errors (413, 404)
        if (!res.body.empty()) return httplib::Server::HandlerResponse::Unhandled;  // ours, already JSON
        fail(res, res.status, httplib::status_message(res.status));
        return httplib::Server::HandlerResponse::Handled;
    });
    http->Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
        res.set_content(dump({{"object", "list"}, {"data", {{{"id", model}, {"object", "model"}}}}}), "application/json");
    });
    http->Post("/v1/completions", [this](const httplib::Request& q, httplib::Response& r) { handle(q, r, false); });
    http->Post("/v1/chat/completions", [this](const httplib::Request& q, httplib::Response& r) { handle(q, r, true); });
}

Server::~Server() = default;

bool Server::listen(const std::string& host, int port) {
    std::thread engine_thread([this] { loop(); });
    const bool ok = http->listen(host, port);  // returns once stopped and every handler is done
    stop();  // a failed bind still stops the engine thread
    engine_thread.join();
    return ok;
}

void Server::wait_until_ready() const { http->wait_until_ready(); }

size_t Server::live_requests() {
    std::lock_guard l(m);
    return n_live;
}

void Server::stop() {
    http->stop();
    {
        std::lock_guard l(m);
        stopping = true;
    }
    cv.notify_one();
}

void Server::loop() {
    std::map<int, std::shared_ptr<Stream>> live;  // engine request id -> its stream
    for (;;) {
        std::deque<std::shared_ptr<Stream>> js, cs;
        {
            std::unique_lock l(m);
            n_live = e.reqs.size();
            // idle: sleep here; with live requests step() blocks on the lanes instead
            cv.wait(l, [&] { return !e.reqs.empty() || !jobs.empty() || !cancels.empty() || stopping; });
            if (stopping && jobs.empty() && e.reqs.empty()) return;
            js.swap(jobs), cs.swap(cancels);
        }
        for (auto& s : js) {
            try {
                std::vector<int> ids = s->chat ? e.tok.encode_chat(s->messages) : e.tok.encode(s->prompt);
                const int n = ids.size();
                if (s->default_max) s->p.max_tokens = std::max(1, std::min(4096, e.ctx_limit() - n));
                s->id = e.submit(std::move(ids), s->p);
                live[s->id] = s;
                s->put([&] { s->prompt_tokens = n, s->started = true; });
            } catch (const std::invalid_argument& x) {
                s->put([&] { s->error = x.what(), s->started = true; });
            }
        }
        for (auto& s : cs)
            if (s->id >= 0 && live.erase(s->id)) e.cancel(s->id);
        std::vector<Event> evs;
        try {
            evs = e.step();
        } catch (const std::exception& x) {  // the rows it aborted get token -1 events from the next step()
            fprintf(stderr, "mllm serve: %s\n", x.what());
        }
        for (const Event& ev : evs) {
            auto it = live.find(ev.req_id);
            if (it == live.end()) continue;
            Stream& s = *it->second;
            bool stopped = false;
            auto pieces = ev.token < 0 ? std::vector<Splitter::Piece>{} : s.split.push(ev.text, ev.done, stopped);
            const std::string finish = ev.token < 0           ? "error"
                                       : stopped || (ev.done && e.tok.eos.count(ev.token)) ? "stop"
                                       : ev.done                                           ? "length"
                                                                                           : "";
            s.put([&] {
                s.completion_tokens += ev.token >= 0;
                s.cached = ev.cached;
                for (auto& p : pieces) s.q.push_back(std::move(p));
                s.finish = finish;
            });
            if (stopped && !ev.done) e.cancel(ev.req_id);
            if (!finish.empty()) live.erase(it);
        }
    }
}

}  // namespace mllm
