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

// an SSE body's data lines, [DONE] last and only once
static std::vector<json> events(const std::string& body) {
    std::vector<json> out;
    bool done = false;
    for (size_t at = 0, nl; (nl = body.find('\n', at)) != std::string::npos; at = nl + 1) {
        const std::string line = body.substr(at, nl - at);
        if (!line.starts_with("data: ")) continue;
        assert(!done);  // nothing after [DONE]
        if (line == "data: [DONE]") done = true;
        else out.push_back(json::parse(line.substr(6)));
    }
    assert(done);
    return out;
}

static std::string run(const std::string& cmd) {
    FILE* f = popen(cmd.c_str(), "r");
    assert(f);
    std::string out;
    char buf[4096];
    for (size_t n; (n = fread(buf, 1, sizeof buf, f));) out.append(buf, n);
    assert(pclose(f) == 0);
    return out;
}

// POST body, read until `until` arrives, then close the connection
static void hang_up(const std::string& body, const std::string& until) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{.sin_family = AF_INET, .sin_port = htons(PORT), .sin_addr = {htonl(INADDR_LOOPBACK)}};
    assert(fd >= 0 && connect(fd, (sockaddr*)&a, sizeof a) == 0);
    const std::string req = "POST /v1/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                            "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    assert(write(fd, req.data(), req.size()) == (ssize_t)req.size());
    std::string got;
    char buf[4096];
    for (ssize_t n; got.find(until) == std::string::npos && (n = read(fd, buf, sizeof buf)) > 0;) got.append(buf, n);
    assert(got.find(until) != std::string::npos);
    close(fd);
}

int main() {
    std::string cli_path = METALLIB;  // build/mllm.metallib -> build/mllm
    cli_path = cli_path.substr(0, cli_path.rfind('/')) + "/mllm";
    std::string want = run(cli_path + " generate " + golden::MODEL + " '" + PROMPT + "' --max-tokens 48 2>/dev/null");
    assert(!want.empty() && want.back() == '\n');
    want.pop_back();  // generate's closing newline

    // one request up to max_ctx (the 15000-token hang-up below); also holds the short ones and their snapshots
    mllm::Engine e(golden::MODEL, std::max<size_t>(256 << 20, mllm::Memory::arena_for(mllm::spec::load(golden::MODEL), 1, 16384)));
    mllm::Server s(e, golden::MODEL);
    bool listened = false;
    std::thread t([&] { listened = s.listen("127.0.0.1", PORT); });
    s.wait_until_ready();
    httplib::Client c("127.0.0.1", PORT);
    c.set_read_timeout(120);

    // the chat template as text: /v1/completions doesn't apply it, generate does
    const std::string chat = std::string("<|im_start|>user\n") + PROMPT + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
    json body = {{"prompt", chat}, {"max_tokens", 48}, {"temperature", 0.0}, {"stream", true},
                 {"stream_options", {{"include_usage", true}}}, {"top_p", nullptr}};  // null = absent
    auto r = c.Post("/v1/completions", body.dump(), "application/json");
    assert(r && r->status == 200);
    std::string text, finish;
    int completion = -1;
    bool done = false;
    for (size_t at = 0, nl; (nl = r->body.find('\n', at)) != std::string::npos; at = nl + 1) {
        const std::string line = r->body.substr(at, nl - at);
        if (!line.starts_with("data: ")) continue;
        assert(!done);  // nothing after [DONE]
        if (line == "data: [DONE]") {
            done = true;
            continue;
        }
        const json d = json::parse(line.substr(6));
        if (d.contains("usage")) completion = d["usage"]["completion_tokens"];
        if (d["choices"].empty()) continue;
        assert(finish.empty());  // finish_reason only on the last token
        text += d["choices"][0]["text"];
        if (!d["choices"][0]["finish_reason"].is_null()) finish = d["choices"][0]["finish_reason"];
    }
    printf("streamed %d tokens, finish %s: %s\n", completion, finish.c_str(), text.c_str());
    assert(done && text == want);
    assert(finish == "length" ? completion == 48 : finish == "stop" && completion > 0 && completion < 48);

    // not streamed: the same text and usage
    auto post = [&](const char* path, json b) {
        auto r = c.Post(path, b.dump(), "application/json");
        assert(r);
        return r;
    };
    body.erase("stream"), body.erase("stream_options");
    r = post("/v1/completions", body);
    assert(r->status == 200);
    json d = json::parse(r->body);
    assert(d["choices"][0]["text"] == want && d["choices"][0]["finish_reason"] == finish &&
           d["usage"]["completion_tokens"] == completion);
    printf("usage: ttft %.1f ms, %.1f tok/s\n", d["usage"]["ttft_ms"].get<double>(),
           d["usage"]["tokens_per_second"].get<double>());
    assert(d["usage"]["ttft_ms"] > 0 && d["usage"]["tokens_per_second"] > 0);

    // chat: the same tokens (generate templates the same message); reasoning is the text before "</think>"
    std::string want_reasoning = want, want_content;
    if (size_t at = want.find("</think>"); at != std::string::npos) {
        want_reasoning = want.substr(0, at);
        want_content = want.substr(at + 8);
        want_content.erase(0, want_content.find_first_not_of('\n'));
    }
    const json msgs = {{{"role", "user"}, {"content", PROMPT}}};
    r = post("/v1/chat/completions", {{"messages", msgs}, {"max_tokens", 48}, {"stream", true}});
    assert(r->status == 200);
    std::string reasoning, content, chat_finish;
    const std::vector<json> evs = events(r->body);
    assert(evs[0]["choices"][0]["delta"]["role"] == "assistant");
    for (const json& d : evs) {
        assert(d["object"] == "chat.completion.chunk" && chat_finish.empty());
        const json& delta = d["choices"][0]["delta"];
        reasoning += delta.value("reasoning_content", ""), content += delta.value("content", "");
        if (!d["choices"][0]["finish_reason"].is_null()) chat_finish = d["choices"][0]["finish_reason"];
    }
    printf("chat: reasoning %zu bytes, content %zu bytes, finish %s\n", reasoning.size(), content.size(),
           chat_finish.c_str());
    assert(reasoning == want_reasoning && content == want_content && chat_finish == finish);
    r = post("/v1/chat/completions", {{"messages", msgs}, {"max_completion_tokens", 48}});
    assert(r->status == 200);
    d = json::parse(r->body);
    const json& msg = d["choices"][0]["message"];
    assert(d["object"] == "chat.completion" && msg["role"] == "assistant" && msg["content"] == want_content &&
           msg.value("reasoning_content", "") == want_reasoning && d["choices"][0]["finish_reason"] == finish);
    // the same with text parts as content
    r = post("/v1/chat/completions", {{"messages", {{{"role", "user"}, {"content", {{{"type", "text"}, {"text", PROMPT}}}}}}},
                                      {"max_tokens", 48}});
    assert(r->status == 200 && json::parse(r->body)["choices"][0]["message"]["content"] == want_content);

    // stop strings: the text ends before the first one, which isn't returned; streamed and not
    const std::string stop = want.substr(want.size() / 2, 4);
    const std::string cut = want.substr(0, want.find(stop));
    for (bool streamed : {false, true}) {
        r = post("/v1/completions", {{"prompt", chat}, {"max_tokens", 48}, {"stop", {"never in it", stop}},
                                     {"stream", streamed}});
        assert(r->status == 200);
        std::string got, why;
        if (!streamed) {
            d = json::parse(r->body);
            got = d["choices"][0]["text"], why = d["choices"][0]["finish_reason"];
        } else {
            for (const json& d : events(r->body)) {
                got += d["choices"][0]["text"];
                if (!d["choices"][0]["finish_reason"].is_null()) why = d["choices"][0]["finish_reason"];
            }
        }
        printf("stop \"%s\" (%s): %zu of %zu bytes, finish %s\n", stop.c_str(), streamed ? "streamed" : "not streamed",
               got.size(), want.size(), why.c_str());
        assert(got == cut && why == "stop");
    }
    r = post("/v1/completions", {{"prompt", chat}, {"max_tokens", 48}, {"stop", "never in it"}});
    assert(json::parse(r->body)["choices"][0]["text"] == want);

    // a repeated prompt: usage reports the prefix cache's tokens (its last full page) the second time
    std::string longp;
    for (int i = 0; i < 20; i++) longp += "The tide comes in and the tide goes out. ";
    const int P = ((int)e.tokenizer.encode(longp).size() - 1) / 32 * 32;
    for (int want_cached : {0, P}) {
        auto u = c.Post("/v1/completions",
                        json({{"prompt", longp}, {"max_tokens", 2}, {"stream", true},
                              {"stream_options", {{"include_usage", true}}}}).dump(), "application/json");
        assert(u && u->status == 200);
        const size_t at = u->body.find("\"usage\"");
        assert(at != std::string::npos);
        const size_t from = u->body.rfind("data: ", at) + 6;
        const json d = json::parse(u->body.substr(from, u->body.find('\n', from) - from));
        assert(P > 0 && d["usage"]["prompt_tokens_details"]["cached_tokens"] == want_cached);
    }
    printf("usage: cached_tokens 0, then %d\n", P);

    for (const std::string& bad : {std::string("{\"prompt\": "), std::string("[1]"),
                                   json({{"prompt", "hi"}, {"max_tokens", 16384}, {"stream", true}}).dump(),
                                   json({{"prompt", "hi"}, {"temperature", "hot"}, {"stream", true}}).dump(),
                                   json({{"prompt", "hi"}, {"max_tokens", 4294967298}, {"stream", true}}).dump(),
                                   json({{"prompt", "hi"}, {"top_p", true}, {"stream", true}}).dump(),
                                   json({{"prompt", "hi"}, {"stop", {"a", "b", "c", "d", "e"}}}).dump(),
                                   json({{"prompt", "hi"}, {"stop", ""}}).dump()}) {
        auto b = c.Post("/v1/completions", bad, "application/json");
        assert(b && b->status == 400 && json::parse(b->body).contains("error"));
    }
    for (const char* bad : {R"({"prompt": "hi"})", R"({"messages": [{"role": "tool", "content": "x"}]})",
                            R"({"messages": []})", R"({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]})"})
        assert(post("/v1/chat/completions", json::parse(bad))->status == 400);
    auto big = c.Post("/v1/completions", json({{"prompt", std::string(2 << 20, 'a')}, {"stream", true}}).dump(),
                      "application/json");
    assert(big && big->status == 413 && json::parse(big->body).contains("error"));

    // hang up after the first chunk of a long request
    body = {{"prompt", "Count to a million."}, {"max_tokens", 15000}, {"stream", true}};
    hang_up(body.dump(), "data: ");
    const auto t0 = std::chrono::steady_clock::now();
    auto sec = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    while (s.live_requests() && sec() < 5) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    printf("hung-up request gone after %.2f s\n", sec());
    assert(s.live_requests() == 0);  // stop() cancels everything, so check before it
    s.stop();
    t.join();
    assert(listened && e.requests.empty());
    printf("PASS\n");
}
