// Server on 127.0.0.1:18007, in this process (an assert can't leave it running): one streamed greedy completion
// equals `mllm generate`'s text, bad requests get 400 / 413 + a JSON error, and a client that hangs up cancels its
// request (else it would keep generating its 15000 tokens).
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

    mllm::Engine e(golden::MODEL, 256 << 20);
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

    // a repeated prompt: usage reports the prefix cache's tokens (its last full page) the second time
    std::string longp;
    for (int i = 0; i < 20; i++) longp += "The tide comes in and the tide goes out. ";
    const int P = ((int)e.tok.encode(longp).size() - 1) / 32 * 32;
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
                                   json({{"prompt", "hi"}, {"top_p", true}, {"stream", true}}).dump()}) {
        auto b = c.Post("/v1/completions", bad, "application/json");
        assert(b && b->status == 400 && json::parse(b->body).contains("error"));
    }
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
    assert(listened && e.reqs.empty());
    printf("PASS\n");
}
