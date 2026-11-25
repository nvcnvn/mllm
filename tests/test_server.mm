// Server on 127.0.0.1:18007, in this process (an assert can't leave it running): one streamed greedy completion
// equals `mllm generate`'s text, and so do the non-streamed one and chat completions (streamed and not; reasoning
// split at "</think>"); stop strings cut the text; bad requests get 400 / 413 + a JSON error, and a client that hangs
// up cancels its request (else it would keep generating its 15000 tokens).
#include <cassert>
#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include <arpa/inet.h>
#include <unistd.h>

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include "golden.h"
#include "server.h"

using json = nlohmann::json;
constexpr int PORT = 18007;
constexpr const char* PROMPT = "Write a haiku about the sea.";

// the chat template as text: /v1/completions doesn't apply it, generate does
static const std::string CHAT_TEXT =
    std::string("<|im_start|>user\n") + PROMPT + "<|im_end|>\n<|im_start|>assistant\n<think>\n";

// a server-sent-events body's data lines, [DONE] last and only once
static std::vector<json> data_events(const std::string& body) {
    std::vector<json> events;
    bool done = false;
    for (size_t at = 0, newline; (newline = body.find('\n', at)) != std::string::npos; at = newline + 1) {
        const std::string line = body.substr(at, newline - at);
        if (!line.starts_with("data: ")) continue;
        assert(!done);  // nothing after [DONE]
        if (line == "data: [DONE]") done = true;
        else events.push_back(json::parse(line.substr(6)));
    }
    assert(done);
    return events;
}

static std::string run_command(const std::string& command) {
    FILE* pipe = popen(command.c_str(), "r");
    assert(pipe);
    std::string output;
    char buffer[4096];
    for (size_t count; (count = fread(buffer, 1, sizeof buffer, pipe));) output.append(buffer, count);
    assert(pclose(pipe) == 0);
    return output;
}

// POST body, read until `until` arrives, then close the connection
static void post_and_hang_up(const std::string& body, const std::string& until) {
    const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in address{.sin_family = AF_INET, .sin_port = htons(PORT), .sin_addr = {htonl(INADDR_LOOPBACK)}};
    assert(socket_fd >= 0 && connect(socket_fd, (sockaddr*)&address, sizeof address) == 0);
    const std::string request = "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                                "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    assert(write(socket_fd, request.data(), request.size()) == (ssize_t)request.size());
    std::string received;
    char buffer[4096];
    for (ssize_t count; received.find(until) == std::string::npos && (count = read(socket_fd, buffer, sizeof buffer)) > 0;)
        received.append(buffer, count);
    assert(received.find(until) != std::string::npos);
    close(socket_fd);
}

static httplib::Result post(httplib::Client& client, const char* path, const json& body) {
    auto response = client.Post(path, body.dump(), "application/json");
    assert(response);
    return response;
}

// what the streamed completion ended with: the other forms must match it
struct Completion {
    std::string finish_reason;
    int completion_tokens = -1;
};

static Completion check_streamed_completion(httplib::Client& client, const json& body, const std::string& expected) {
    auto response = client.Post("/v1/completions", body.dump(), "application/json");
    assert(response && response->status == 200);
    std::string text;
    Completion completion;
    bool done = false;
    const std::string& stream = response->body;
    for (size_t at = 0, newline; (newline = stream.find('\n', at)) != std::string::npos; at = newline + 1) {
        const std::string line = stream.substr(at, newline - at);
        if (!line.starts_with("data: ")) continue;
        assert(!done);  // nothing after [DONE]
        if (line == "data: [DONE]") {
            done = true;
            continue;
        }
        const json chunk = json::parse(line.substr(6));
        if (chunk.contains("usage")) completion.completion_tokens = chunk["usage"]["completion_tokens"];
        if (chunk["choices"].empty()) continue;
        assert(completion.finish_reason.empty());  // finish_reason only on the last token
        text += chunk["choices"][0]["text"];
        if (!chunk["choices"][0]["finish_reason"].is_null())
            completion.finish_reason = chunk["choices"][0]["finish_reason"];
    }
    printf("streamed %d tokens, finish %s: %s\n", completion.completion_tokens, completion.finish_reason.c_str(),
           text.c_str());
    assert(done && text == expected);
    assert(completion.finish_reason == "length"
               ? completion.completion_tokens == 48
               : completion.finish_reason == "stop" && completion.completion_tokens > 0 &&
                     completion.completion_tokens < 48);
    return completion;
}

// not streamed: the same text and usage
static void check_non_streamed_completion(httplib::Client& client, json body, const std::string& expected,
                                          const Completion& streamed) {
    body.erase("stream"), body.erase("stream_options");
    auto response = post(client, "/v1/completions", body);
    assert(response->status == 200);
    json reply = json::parse(response->body);
    assert(reply["choices"][0]["text"] == expected && reply["choices"][0]["finish_reason"] == streamed.finish_reason &&
           reply["usage"]["completion_tokens"] == streamed.completion_tokens);
    printf("usage: ttft %.1f ms, %.1f tok/s\n", reply["usage"]["ttft_ms"].get<double>(),
           reply["usage"]["tokens_per_second"].get<double>());
    assert(reply["usage"]["ttft_ms"] > 0 && reply["usage"]["tokens_per_second"] > 0);
}

// chat: the same tokens (generate templates the same message); reasoning is the text before "</think>"
static void check_chat_completions(httplib::Client& client, const std::string& expected,
                                   const std::string& finish_reason) {
    std::string expected_reasoning = expected, expected_content;
    if (size_t at = expected.find("</think>"); at != std::string::npos) {
        expected_reasoning = expected.substr(0, at);
        expected_content = expected.substr(at + 8);
        expected_content.erase(0, expected_content.find_first_not_of('\n'));
    }
    const json messages = {{{"role", "user"}, {"content", PROMPT}}};
    auto response = post(client, "/v1/chat/completions", {{"messages", messages}, {"max_tokens", 48}, {"stream", true}});
    assert(response->status == 200);
    std::string reasoning, content, chat_finish_reason;
    const std::vector<json> chunks = data_events(response->body);
    assert(chunks[0]["choices"][0]["delta"]["role"] == "assistant");
    for (const json& chunk : chunks) {
        assert(chunk["object"] == "chat.completion.chunk" && chat_finish_reason.empty());
        const json& delta = chunk["choices"][0]["delta"];
        reasoning += delta.value("reasoning_content", ""), content += delta.value("content", "");
        if (!chunk["choices"][0]["finish_reason"].is_null()) chat_finish_reason = chunk["choices"][0]["finish_reason"];
    }
    printf("chat: reasoning %zu bytes, content %zu bytes, finish %s\n", reasoning.size(), content.size(),
           chat_finish_reason.c_str());
    assert(reasoning == expected_reasoning && content == expected_content && chat_finish_reason == finish_reason);
    response = post(client, "/v1/chat/completions", {{"messages", messages}, {"max_completion_tokens", 48}});
    assert(response->status == 200);
    json reply = json::parse(response->body);
    const json& message = reply["choices"][0]["message"];
    assert(reply["object"] == "chat.completion" && message["role"] == "assistant" &&
           message["content"] == expected_content &&
           message.value("reasoning_content", "") == expected_reasoning &&
           reply["choices"][0]["finish_reason"] == finish_reason);
    // the same with text parts as content
    response = post(client, "/v1/chat/completions",
                    {{"messages", {{{"role", "user"}, {"content", {{{"type", "text"}, {"text", PROMPT}}}}}}},
                     {"max_tokens", 48}});
    assert(response->status == 200 &&
           json::parse(response->body)["choices"][0]["message"]["content"] == expected_content);
}

// the text and finish reason of a stop-string completion, streamed or not
static void read_stopped_completion(const std::string& body, bool streamed, std::string& text,
                                    std::string& finish_reason) {
    if (!streamed) {
        json reply = json::parse(body);
        text = reply["choices"][0]["text"], finish_reason = reply["choices"][0]["finish_reason"];
        return;
    }
    for (const json& chunk : data_events(body)) {
        text += chunk["choices"][0]["text"];
        if (!chunk["choices"][0]["finish_reason"].is_null()) finish_reason = chunk["choices"][0]["finish_reason"];
    }
}

// stop strings: the text ends before the first one, which isn't returned; streamed and not
static void check_stop_strings(httplib::Client& client, const std::string& expected) {
    const std::string stop = expected.substr(expected.size() / 2, 4);
    const std::string cut = expected.substr(0, expected.find(stop));
    for (bool streamed : {false, true}) {
        auto response = post(client, "/v1/completions",
                             {{"prompt", CHAT_TEXT}, {"max_tokens", 48}, {"stop", {"never in it", stop}},
                              {"stream", streamed}});
        assert(response->status == 200);
        std::string text, finish_reason;
        read_stopped_completion(response->body, streamed, text, finish_reason);
        printf("stop \"%s\" (%s): %zu of %zu bytes, finish %s\n", stop.c_str(), streamed ? "streamed" : "not streamed",
               text.size(), expected.size(), finish_reason.c_str());
        assert(text == cut && finish_reason == "stop");
    }
    auto response = post(client, "/v1/completions", {{"prompt", CHAT_TEXT}, {"max_tokens", 48}, {"stop", "never in it"}});
    assert(json::parse(response->body)["choices"][0]["text"] == expected);
}

// a repeated prompt: usage reports the prefix cache's tokens (its last full page) the second time
static void check_cached_tokens_in_usage(httplib::Client& client, mllm::Engine& engine) {
    std::string long_prompt;
    for (int i = 0; i < 20; i++) long_prompt += "The tide comes in and the tide goes out. ";
    const int cached_prefix_tokens = ((int)engine.tokenizer.encode(long_prompt).size() - 1) / 32 * 32;
    for (int expected_cached : {0, cached_prefix_tokens}) {
        auto response = client.Post("/v1/completions",
                                    json({{"prompt", long_prompt}, {"max_tokens", 2}, {"stream", true},
                                          {"stream_options", {{"include_usage", true}}}}).dump(),
                                    "application/json");
        assert(response && response->status == 200);
        const size_t at = response->body.find("\"usage\"");
        assert(at != std::string::npos);
        const size_t from = response->body.rfind("data: ", at) + 6;
        const json usage_chunk = json::parse(response->body.substr(from, response->body.find('\n', from) - from));
        assert(cached_prefix_tokens > 0 &&
               usage_chunk["usage"]["prompt_tokens_details"]["cached_tokens"] == expected_cached);
    }
    printf("usage: cached_tokens 0, then %d\n", cached_prefix_tokens);
}

// bad requests get 400 / 413 + a JSON error
static void check_bad_requests(httplib::Client& client) {
    for (const std::string& bad : {std::string("{\"prompt\": "), std::string("[1]"),
                                   json({{"prompt", "hi"}, {"max_tokens", 16384}, {"stream", true}}).dump(),
                                   json({{"prompt", "hi"}, {"temperature", "hot"}, {"stream", true}}).dump(),
                                   json({{"prompt", "hi"}, {"max_tokens", 4294967298}, {"stream", true}}).dump(),
                                   json({{"prompt", "hi"}, {"top_p", true}, {"stream", true}}).dump(),
                                   json({{"prompt", "hi"}, {"stop", {"a", "b", "c", "d", "e"}}}).dump(),
                                   json({{"prompt", "hi"}, {"stop", ""}}).dump()}) {
        auto response = client.Post("/v1/completions", bad, "application/json");
        assert(response && response->status == 400 && json::parse(response->body).contains("error"));
    }
    for (const char* bad : {R"({"prompt": "hi"})", R"({"messages": [{"role": "tool", "content": "x"}]})",
                            R"({"messages": []})", R"({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})"})
        assert(post(client, "/v1/chat/completions", json::parse(bad))->status == 400);
    auto too_big = client.Post("/v1/completions", json({{"prompt", std::string(2 << 20, 'a')}, {"stream", true}}).dump(),
                               "application/json");
    assert(too_big && too_big->status == 413 && json::parse(too_big->body).contains("error"));
}

// hang up after the first chunk of a long request: the server cancels it
static void check_hang_up_cancels(mllm::Server& server) {
    json body = {{"prompt", "Count to a million."}, {"max_tokens", 15000}, {"stream", true}};
    post_and_hang_up(body.dump(), "data: ");
    const auto hung_up_at = std::chrono::steady_clock::now();
    auto seconds_since = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - hung_up_at).count();
    };
    while (server.live_requests() && seconds_since() < 5) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    printf("hung-up request gone after %.2f s\n", seconds_since());
    assert(server.live_requests() == 0);  // stop() cancels everything, so check before it
}

int main() {
    std::string cli_path = METALLIB;  // build/mllm.metallib -> build/mllm
    cli_path = cli_path.substr(0, cli_path.rfind('/')) + "/mllm";
    std::string expected =
        run_command(cli_path + " generate " + golden::MODEL + " '" + PROMPT + "' --max-tokens 48 2>/dev/null");
    assert(!expected.empty() && expected.back() == '\n');
    expected.pop_back();  // generate's closing newline

    // one request up to max_context (the 15000-token hang-up below); also holds the short ones and their snapshots
    mllm::Engine engine(golden::MODEL, std::max<size_t>(256 << 20, mllm::Memory::arena_for(
                                                                       mllm::spec::load(golden::MODEL), 1, 16384)));
    mllm::Server server(engine, golden::MODEL);
    bool listened = false;
    std::thread server_thread([&] { listened = server.listen("127.0.0.1", PORT); });
    server.wait_until_ready();
    httplib::Client client("127.0.0.1", PORT);
    client.set_read_timeout(120);

    const json body = {{"prompt", CHAT_TEXT}, {"max_tokens", 48}, {"temperature", 0.0}, {"stream", true},
                       {"stream_options", {{"include_usage", true}}}, {"top_p", nullptr}};  // null = absent
    const Completion streamed = check_streamed_completion(client, body, expected);
    check_non_streamed_completion(client, body, expected, streamed);
    check_chat_completions(client, expected, streamed.finish_reason);
    check_stop_strings(client, expected);
    check_cached_tokens_in_usage(client, engine);
    check_bad_requests(client);
    check_hang_up_cancels(server);
    server.stop();
    server_thread.join();
    assert(listened && engine.requests.empty());
    printf("PASS\n");
}
