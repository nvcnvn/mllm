#include "server.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <cstdio>
#include <ctime>
#include <map>
#include <limits>
#include <stdexcept>
#include <thread>

namespace mllm {

using json = nlohmann::json;

struct Server::Stream {
    std::string prompt;
    GenParams p;
    int id = -1;  // engine request id, set before started

    std::mutex m;  // guards the rest
    std::condition_variable cv;
    bool started = false;  // submitted (prompt_tokens set) or rejected (error set)
    int prompt_tokens = 0;
    std::string error;
    std::deque<Event> q;
    bool eos = false;  // the last event's token is EOS (set with it)

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
    http->Post("/v1/completions", [this](const httplib::Request& req, httplib::Response& res) {
        auto s = std::make_shared<Stream>();
        bool usage = false;
        // Reads prompt, max_tokens, temperature, top_p, stream, stream_options.include_usage; null = absent (as
        // openai-python sends). ponytail: other fields (stop, n, echo, logprobs, ...) are ignored, not rejected.
        try {
            json j = json::parse(req.body);
            if (!j.is_object()) return fail(res, 400, "need a JSON object");
            for (auto it = j.begin(); it != j.end();) it = it->is_null() ? j.erase(it) : std::next(it);
            if (!j.contains("prompt") || !j["prompt"].is_string()) return fail(res, 400, "need a string prompt");
            if (!j.value("stream", false)) return fail(res, 400, "only stream: true is supported");
            if (j.contains("max_tokens") && !(j["max_tokens"].is_number_integer() && j["max_tokens"] >= 1 &&
                                              j["max_tokens"] <= std::numeric_limits<int>::max()))
                return fail(res, 400, "max_tokens: need an integer >= 1");
            for (const char* k : {"temperature", "top_p"})
                if (j.contains(k) && !j[k].is_number()) return fail(res, 400, std::string(k) + ": need a number");
            s->prompt = j["prompt"];
            s->p = {(int)j.value("max_tokens", (int64_t)256), j.value("temperature", 0.0f), j.value("top_p", 0.0f)};
            const json opts = j.value("stream_options", json::object());
            usage = opts.is_object() && opts.value("include_usage", false);
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
        const json head = {{"id", "cmpl-" + std::to_string(s->id)}, {"object", "text_completion"},
                           {"created", (long)std::time(nullptr)}, {"model", model}};
        auto completion_tokens = std::make_shared<int>(0);
        res.set_chunked_content_provider(
            "text/event-stream",
            [s, head, usage, completion_tokens](size_t, httplib::DataSink& sink) {
                std::deque<Event> evs;
                bool eos;
                {
                    // wake every 100 ms with nothing to send: httplib checks the peer between calls, so a client
                    // that hung up before its first token is cancelled now, not when that token comes
                    std::unique_lock l(s->m);
                    if (!s->cv.wait_for(l, std::chrono::milliseconds(100), [&] { return !s->q.empty(); })) return true;
                    evs.swap(s->q);
                    eos = s->eos;
                }
                auto send = [&](const json& j) {
                    const std::string line = "data: " + dump(j) + "\n\n";
                    return sink.write(line.data(), line.size());
                };
                for (const Event& ev : evs) {
                    json j = head;
                    if (ev.token < 0) j["error"] = {{"message", "aborted: a lane failed"}};
                    else {
                        ++*completion_tokens;
                        j["choices"] = {{{"index", 0}, {"text", ev.text}, {"logprobs", nullptr},
                                         {"finish_reason", ev.done ? json(eos ? "stop" : "length") : json(nullptr)}}};
                    }
                    if (!send(j)) return false;
                    if (!ev.done) continue;
                    if (usage) {
                        j = head;
                        j["choices"] = json::array();
                        j["usage"] = {{"prompt_tokens", s->prompt_tokens}, {"completion_tokens", *completion_tokens},
                                      {"total_tokens", s->prompt_tokens + *completion_tokens}};
                        if (!send(j)) return false;
                    }
                    const std::string end = "data: [DONE]\n\n";
                    if (!sink.write(end.data(), end.size())) return false;
                    sink.done();
                    return true;
                }
                return true;
            },
            [this, s](bool success) {  // the stream ended; not cleanly = the client went away
                if (success) return;
                {
                    std::lock_guard l(m);
                    cancels.push_back(s);
                }
                cv.notify_one();
            });
    });
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
                std::vector<int> ids = e.tok.encode(s->prompt);
                const int n = ids.size();
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
        for (Event& ev : evs) {
            auto it = live.find(ev.req_id);
            if (it == live.end()) continue;
            const bool done = ev.done;
            const bool eos = done && e.tok.eos.count(ev.token);
            it->second->put([&] {
                it->second->eos = eos;
                it->second->q.push_back(std::move(ev));
            });
            if (done) live.erase(it);
        }
    }
}

}  // namespace mllm
