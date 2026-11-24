# A model, as this talk sees it: pure matmul blocks around attention, the only part with memory.

node "tokens" {
  kind  = "terminal"
  label = "tokens"
}

node "dense0" {
  kind  = "service"
  label = "dense\npure matmuls"
}

node "attn0" {
  kind  = "service"
  label = "attn"
}



node "more" {
  kind  = "node"
  label = "… the same,\n24–64 layers"
}

node "next" {
  kind  = "terminal"
  label = "next token"
}

node "memory" {
  kind  = "storage"
  label = "per-request memory\nKV (grows) + state (fixed)"
}

edge "t-d0" {
  from = node.tokens
  to   = node.dense0
}

edge "d0-a0" {
  from = node.dense0
  to   = node.attn0
}




edge "a0-more" {
  from = node.attn0
  to   = node.more
}

edge "more-next" {
  from = node.more
  to   = node.next
}

edge "a0-mem" {
  from  = node.attn0
  to    = node.memory
  kind  = "async"
  label = "read + append"
}

