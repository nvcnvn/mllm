#include "server.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <ctime>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>

namespace mllm {

using json = nlohmann::json;

static double seconds(std::chrono::steady_clock::duration duration) {
    return std::chrono::duration<double>(duration).count();
}

// The longest k > at_least (else at_least) such that text's last k bytes are marker's first k; k < marker's size
static size_t marker_start_at_end(const std::string& text, const std::string& marker, size_t at_least) {
    for (size_t k = std::min(marker.size() - 1, text.size()); k > at_least; k--)
        if (text.compare(text.size() - k, k, marker, 0, k) == 0) return k;
    return at_least;
}

// Generated text with the markers cut out: stop strings end it, and for chat "</think>" ends the reasoning (the
// template opens "<think>\n", so a reply starts inside it). A tail that could be the start of a marker waits for the
// next text. Engine thread only.
struct Splitter {
    std::vector<std::string> stops;
    bool reasoning = false;        // chat: text so far is reasoning_content
    bool content_started = false;  // any content emitted (its leading newlines are dropped)
    std::string held;

    struct Piece {
        std::string text;
        bool reasoning;
    };
    // the text to emit now; stopped = a stop string was found (it and the rest are dropped)
    std::vector<Piece> push(const std::string& text, bool flush, bool& stopped) {
        std::vector<Piece> out;
        held += text;
        auto emit = [&](std::string piece) {
            if (!reasoning && !content_started) piece.erase(0, piece.find_first_not_of('\n'));  // npos clears
            if (piece.empty()) return;
            content_started |= !reasoning;
            out.push_back({std::move(piece), reasoning});
        };
        for (;;) {  // the first marker in held: emit the text before it, then end (stop) or switch to content
            size_t marker_at = std::string::npos, marker_size = 0;
            bool think_end = false;
            for (const std::string& stop : stops)
                if (size_t at = held.find(stop); at < marker_at) marker_at = at, marker_size = stop.size();
            if (reasoning)
                if (size_t at = held.find("</think>"); at < marker_at) marker_at = at, marker_size = 8, think_end = true;
            if (marker_at == std::string::npos) break;
            emit(held.substr(0, marker_at));
            held.erase(0, marker_at + marker_size);
            if (!think_end) return stopped = true, held.clear(), out;
            reasoning = false;
        }
        size_t keep = 0;  // the longest tail that is a proper prefix of a marker
        if (!flush) {
            for (const std::string& stop : stops) keep = marker_start_at_end(held, stop, keep);
            if (reasoning) keep = marker_start_at_end(held, "</think>", keep);
        }
        emit(held.substr(0, held.size() - keep));
        held.erase(0, held.size() - keep);
        return out;
    }
};

struct Server::Stream {
    bool chat = false;
    std::string prompt;
    std::vector<Message> messages;
    GenParams params;
    bool default_max_tokens = false;  // chat without max_tokens: up to 4096, within max_ctx
    std::chrono::steady_clock::time_point arrived = std::chrono::steady_clock::now(), first_token_at, last_token_at;
    Splitter splitter;    // engine thread only
    int request_id = -1;  // engine request id, set before started

    std::mutex mutex;  // guards the rest
    std::condition_variable changed;
    bool started = false;  // submitted (prompt_tokens set) or rejected (error set)
    int prompt_tokens = 0, completion_tokens = 0, cached_tokens = 0;
    std::string error;
    std::deque<Splitter::Piece> pieces;
    std::string finish_reason;  // set last: stop | length | error

    void update(auto&& change) {
        {
            std::lock_guard lock(mutex);
            change();
        }
        changed.notify_one();
    }
};

static std::string dump(const json& value) { return value.dump(-1, ' ', false, json::error_handler_t::replace); }

// prompt bytes: the engine thread tokenizes (~170 ms per MB), stalling every stream meanwhile; max_ctx 16384 tokens
// of text is well under this
constexpr size_t MAX_BODY_BYTES = 1 << 20;

static void send_error(httplib::Response& response, int status, const std::string& message) {
    response.status = status;
    response.set_content(dump({{"error", {{"message", message}, {"type", "invalid_request_error"}}}}), "application/json");
}

// A message's content (a string, or an array of text parts) -> text; an error message, or "" when it is fine
static std::string message_text(const json& content, std::string& text) {
    if (content.is_string()) {
        text = content;
        return "";
    }
    if (content.is_null()) return "";
    if (!content.is_array()) return "messages: content must be a string or text parts";
    for (const json& part : content) {
        if (!(part.is_object() && part.value("type", "") == "text" && part.contains("text") && part["text"].is_string()))
            return "messages: content parts must be text";
        text += part["text"].get<std::string>();
    }
    return "";
}

// The request's fields into stream; an error message, or "" when it is fine. null = absent (as openai-python sends).
// ponytail: other fields (n, echo, logprobs, tools, ...) are ignored, not rejected.
static std::string parse_request(json& body, Server::Stream& stream, bool& streamed, bool& include_usage) {
    if (!body.is_object()) return "need a JSON object";
    for (auto field = body.begin(); field != body.end();) field = field->is_null() ? body.erase(field) : std::next(field);
    if (!stream.chat && !(body.contains("prompt") && body["prompt"].is_string())) return "need a string prompt";
    if (stream.chat) {
        if (!(body.contains("messages") && body["messages"].is_array())) return "need a messages array";
        for (const json& message : body["messages"]) {
            if (!(message.is_object() && message.contains("role") && message["role"].is_string()))
                return "messages: need a role";
            std::string text;
            if (std::string err = message_text(message.value("content", json("")), text); !err.empty()) return err;
            stream.messages.push_back({message["role"], text});
        }
    }
    const char* max_key = body.contains("max_completion_tokens") ? "max_completion_tokens" : "max_tokens";
    if (body.contains(max_key) &&
        !(body[max_key].is_number_integer() && body[max_key] >= 1 && body[max_key] <= std::numeric_limits<int>::max()))
        return std::string(max_key) + ": need an integer >= 1";
    for (const char* key : {"temperature", "top_p"})
        if (body.contains(key) && !body[key].is_number()) return std::string(key) + ": need a number";
    if (body.contains("stop")) {
        json stops = body["stop"].is_string() ? json::array({body["stop"]}) : body["stop"];
        if (!stops.is_array() || stops.size() > 4) return "stop: need a string or up to 4 strings";
        for (const json& stop : stops) {
            if (!stop.is_string() || stop.get<std::string>().empty()) return "stop: need non-empty strings";
            stream.splitter.stops.push_back(stop);
        }
    }
    if (!stream.chat) stream.prompt = body["prompt"];
    stream.default_max_tokens = stream.chat && !body.contains(max_key);
    stream.params = {(int)body.value(max_key, (int64_t)256), body.value("temperature", 0.0f), body.value("top_p", 0.0f)};
    stream.splitter.reasoning = stream.chat;
    streamed = body.value("stream", false);
    const json options = body.value("stream_options", json::object());
    include_usage = options.is_object() && options.value("include_usage", false);
    return "";
}

// OpenAI's fields, plus ttft_ms (arrival to first token) and tokens_per_second (after the first token)
static json usage_json(const Server::Stream& stream) {
    const int tokens = stream.completion_tokens;
    return json{{"prompt_tokens", stream.prompt_tokens},
                {"completion_tokens", tokens},
                {"total_tokens", stream.prompt_tokens + tokens},
                {"prompt_tokens_details", {{"cached_tokens", stream.cached_tokens}}},
                {"ttft_ms", tokens ? std::round(seconds(stream.first_token_at - stream.arrived) * 1e4) / 10 : 0.0},
                {"tokens_per_second",
                 tokens > 1 ? std::round((tokens - 1) / seconds(stream.last_token_at - stream.first_token_at) * 10) / 10
                            : 0.0}};
}

// One streamed choice: text (completions) or a content / reasoning_content delta (chat)
static json choice_json(bool chat, const std::string& text, bool reasoning, const json& finish_reason) {
    if (!chat) return json{{"index", 0}, {"text", text}, {"logprobs", nullptr}, {"finish_reason", finish_reason}};
    json delta = json::object();
    if (!text.empty()) delta[reasoning ? "reasoning_content" : "content"] = text;
    return json{{"index", 0}, {"delta", delta}, {"logprobs", nullptr}, {"finish_reason", finish_reason}};
}

// One call of the streamed reply's content provider: sends what the stream has (the role first, for chat), then the
// finish and usage chunks and [DONE] once it is finished. false: the client went away.
static bool send_stream_events(Server::Stream& stream, const json& head, bool chat, bool include_usage, bool& role_sent,
                               httplib::DataSink& sink) {
    std::deque<Splitter::Piece> pieces;
    std::string finish_reason;
    {
        // wake every 100 ms with nothing to send: httplib checks the peer between calls, so a client
        // that hung up before its first token is cancelled now, not when that token comes
        std::unique_lock lock(stream.mutex);
        if (!stream.changed.wait_for(lock, std::chrono::milliseconds(100),
                                     [&] { return !stream.pieces.empty() || !stream.finish_reason.empty(); }))
            return true;
        pieces.swap(stream.pieces);
        finish_reason = stream.finish_reason;
    }
    auto send = [&](const json& chunk) {
        const std::string line = "data: " + dump(chunk) + "\n\n";
        return sink.write(line.data(), line.size());
    };
    if (chat && !std::exchange(role_sent, true)) {  // OpenAI's first chunk: the role
        json chunk = head;
        chunk["choices"] = {{{"index", 0}, {"delta", {{"role", "assistant"}, {"content", ""}}},
                             {"logprobs", nullptr}, {"finish_reason", nullptr}}};
        if (!send(chunk)) return false;
    }
    for (const auto& [text, reasoning] : pieces) {
        json chunk = head;
        chunk["choices"] = {choice_json(chat, text, reasoning, nullptr)};
        if (!send(chunk)) return false;
    }
    if (finish_reason.empty()) return true;
    json chunk = head;
    if (finish_reason == "error") chunk["error"] = {{"message", "aborted: a lane failed"}};
    else chunk["choices"] = {choice_json(chat, "", false, finish_reason)};
    if (!send(chunk)) return false;
    if (include_usage) {
        chunk = head;
        chunk["choices"] = json::array();
        chunk["usage"] = usage_json(stream);
        if (!send(chunk)) return false;
    }
    const std::string end = "data: [DONE]\n\n";
    if (!sink.write(end.data(), end.size())) return false;
    sink.done();
    return true;
}

void Server::queue_cancel(const std::shared_ptr<Stream>& stream) {
    {
        std::lock_guard lock(mutex);
        cancelled_streams.push_back(stream);
    }
    wake_engine.notify_one();
}

void Server::handle_request(const httplib::Request& request, httplib::Response& response, bool chat) {
    auto stream = std::make_shared<Stream>();
    stream->chat = chat;
    bool streamed = false, include_usage = false;
    try {
        json body = json::parse(request.body);
        if (std::string err = parse_request(body, *stream, streamed, include_usage); !err.empty())
            return send_error(response, 400, err);
    } catch (const json::exception& x) {
        return send_error(response, 400, x.what());
    }
    {
        std::lock_guard lock(mutex);
        if (stopping) return send_error(response, 503, "shutting down");
        new_streams.push_back(stream);
    }
    wake_engine.notify_one();
    {
        std::unique_lock lock(stream->mutex);
        stream->changed.wait(lock, [&] { return stream->started; });
        if (!stream->error.empty()) return send_error(response, 400, stream->error);
    }
    const json head = {{"id", (chat ? "chatcmpl-" : "cmpl-") + std::to_string(stream->request_id)},
                       {"object", chat ? (streamed ? "chat.completion.chunk" : "chat.completion") : "text_completion"},
                       {"created", (long)std::time(nullptr)},
                       {"model", model_name}};

    if (!streamed) {
        std::string text, reasoning;
        std::unique_lock lock(stream->mutex);
        // wake every 100 ms: a client that hung up is cancelled, not left generating
        while (!stream->changed.wait_for(lock, std::chrono::milliseconds(100),
                                         [&] { return !stream->finish_reason.empty(); }))
            if (request.is_connection_closed()) {
                lock.unlock();
                return queue_cancel(stream);
            }
        if (stream->finish_reason == "error") return send_error(response, 500, "aborted: a lane failed");
        for (const auto& [piece, is_reasoning] : stream->pieces) (is_reasoning ? reasoning : text) += piece;
        json reply = head;
        json choice = {{"index", 0}, {"logprobs", nullptr}, {"finish_reason", stream->finish_reason}};
        if (chat) {
            choice["message"] = {{"role", "assistant"}, {"content", text}};
            if (!reasoning.empty()) choice["message"]["reasoning_content"] = reasoning;
        } else {
            choice["text"] = text;
        }
        reply["choices"] = {choice};
        reply["usage"] = usage_json(*stream);
        response.set_content(dump(reply), "application/json");
        return;
    }

    auto role_sent = std::make_shared<bool>(false);
    response.set_chunked_content_provider(
        "text/event-stream",
        [stream, head, include_usage, chat, role_sent](size_t, httplib::DataSink& sink) {
            return send_stream_events(*stream, head, chat, include_usage, *role_sent, sink);
        },
        [this, stream](bool success) {  // the stream ended; not cleanly = the client went away
            if (!success) queue_cancel(stream);
        });
}

Server::Server(Engine& engine, std::string model_name)
    : engine(engine), model_name(std::move(model_name)), http(std::make_unique<httplib::Server>()) {
    // httplib's default is SO_REUSEPORT on macOS: a second server on a busy port would bind and share it
    http->set_socket_options([](socket_t sock) { httplib::set_socket_opt(sock, SOL_SOCKET, SO_REUSEADDR, 1); });
    http->set_payload_max_length(MAX_BODY_BYTES);
    // httplib's own errors (413, 404)
    http->set_error_handler([](const httplib::Request&, httplib::Response& response) {
        if (!response.body.empty()) return httplib::Server::HandlerResponse::Unhandled;  // ours, already JSON
        send_error(response, response.status, httplib::status_message(response.status));
        return httplib::Server::HandlerResponse::Handled;
    });
    http->Get("/v1/models", [this](const httplib::Request&, httplib::Response& response) {
        response.set_content(dump({{"object", "list"}, {"data", {{{"id", this->model_name}, {"object", "model"}}}}}),
                             "application/json");
    });
    http->Post("/v1/completions", [this](const httplib::Request& request, httplib::Response& response) {
        handle_request(request, response, false);
    });
    http->Post("/v1/chat/completions", [this](const httplib::Request& request, httplib::Response& response) {
        handle_request(request, response, true);
    });
}

Server::~Server() = default;

bool Server::listen(const std::string& host, int port) {
    std::thread engine_thread([this] { run_engine(); });
    const bool ok = http->listen(host, port);  // returns once stopped and every handler is done
    stop();  // a failed bind still stops the engine thread
    engine_thread.join();
    return ok;
}

void Server::wait_until_ready() const { http->wait_until_ready(); }

size_t Server::live_requests() {
    std::lock_guard lock(mutex);
    return live_count;
}

void Server::stop() {
    http->stop();
    {
        std::lock_guard lock(mutex);
        stopping = true;
    }
    wake_engine.notify_one();
}

void Server::run_engine() {
    for (;;) {
        std::deque<std::shared_ptr<Stream>> to_start, to_cancel;
        {
            std::unique_lock lock(mutex);
            live_count = engine.requests.size();
            // idle: sleep here; with live requests step() blocks on the lanes instead
            wake_engine.wait(lock, [&] {
                return !engine.requests.empty() || !new_streams.empty() || !cancelled_streams.empty() || stopping;
            });
            if (stopping && new_streams.empty() && engine.requests.empty()) return;
            to_start.swap(new_streams), to_cancel.swap(cancelled_streams);
        }
        start_streams(to_start);
        cancel_streams(to_cancel);
        std::vector<Event> events;
        try {
            events = engine.step();
        } catch (const std::exception& x) {  // the rows it aborted get token -1 events from the next step()
            fprintf(stderr, "mllm serve: %s\n", x.what());
        }
        for (const Event& event : events) deliver(event);
    }
}

void Server::start_streams(const std::deque<std::shared_ptr<Stream>>& streams) {
    for (const auto& stream : streams) {
        try {
            std::vector<int> tokens = stream->chat ? engine.tokenizer.encode_chat(stream->messages)
                                                   : engine.tokenizer.encode(stream->prompt);
            const int prompt_tokens = tokens.size();
            if (stream->default_max_tokens)
                stream->params.max_tokens = std::max(1, std::min(4096, engine.context_limit() - prompt_tokens));
            stream->request_id = engine.submit(std::move(tokens), stream->params);
            live_streams[stream->request_id] = stream;
            stream->update([&] { stream->prompt_tokens = prompt_tokens, stream->started = true; });
        } catch (const std::invalid_argument& x) {
            stream->update([&] { stream->error = x.what(), stream->started = true; });
        }
    }
}

void Server::cancel_streams(const std::deque<std::shared_ptr<Stream>>& streams) {
    for (const auto& stream : streams)
        if (stream->request_id >= 0 && live_streams.erase(stream->request_id)) engine.cancel(stream->request_id);
}

static void log_finished(Server::Stream& stream, const std::string& finish_reason) {
    std::lock_guard lock(stream.mutex);  // handler threads only read these
    const int tokens = stream.completion_tokens;
    fprintf(stderr, "mllm serve: request %d: prompt %d tokens, %d cached, first token %.3f s; %d tokens at "
            "%.1f tok/s, %s\n", stream.request_id, stream.prompt_tokens, stream.cached_tokens,
            tokens ? seconds(stream.first_token_at - stream.arrived) : 0.0, tokens,
            tokens > 1 ? (tokens - 1) / seconds(stream.last_token_at - stream.first_token_at) : 0.0,
            finish_reason.c_str());
}

void Server::deliver(const Event& event) {
    auto found = live_streams.find(event.req_id);
    if (found == live_streams.end()) return;
    Stream& stream = *found->second;
    bool stopped = false;
    auto pieces = event.token < 0 ? std::vector<Splitter::Piece>{} : stream.splitter.push(event.text, event.done, stopped);
    const std::string finish_reason = event.token < 0                                                  ? "error"
                                      : stopped || (event.done && engine.tokenizer.eos.count(event.token)) ? "stop"
                                      : event.done                                                      ? "length"
                                                                                                        : "";
    const auto now = std::chrono::steady_clock::now();
    stream.update([&] {
        if (event.token >= 0 && !stream.completion_tokens++) stream.first_token_at = now;
        stream.last_token_at = now;
        stream.cached_tokens = event.cached;
        for (auto& piece : pieces) stream.pieces.push_back(std::move(piece));
        stream.finish_reason = finish_reason;
    });
    if (stopped && !event.done) engine.cancel(event.req_id);
    if (finish_reason.empty()) return;
    log_finished(stream, finish_reason);
    live_streams.erase(found);
}

}  // namespace mllm
