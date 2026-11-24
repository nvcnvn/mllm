# The prefix cache: one node per full 32-token page, keyed by chained hashes; ★ nodes hold a state snapshot.

node "root" {
  kind  = "initial"
}

node "p0" {
  kind  = "node"
  label = "page 0\nh0 = hash(root, tokens 0–31)"
}

node "p1" {
  kind  = "node"
  label = "page 1\nh1 = hash(h0, tokens 32–63)"
}

node "p2" {
  kind  = "node"
  label = "page 2 ★\nconversation A"
}

node "p2b" {
  kind  = "node"
  label = "page 2' ★\nconversation B"
}

edge "r-p0" {
  from = node.root
  to   = node.p0
}

edge "p0-p1" {
  from = node.p0
  to   = node.p1
}

edge "p1-p2" {
  from = node.p1
  to   = node.p2
}

edge "p1-p2b" {
  from = node.p1
  to   = node.p2b
}
