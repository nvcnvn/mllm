# One arena unit's life (src/memory.mm Arena). free_units counts free + pending; cache-only units are evictable.

node "start" {
  kind = "initial"
}

node "free" {
  kind  = "simple-state"
  label = "free\nbit set in the bitmap"
}

node "live" {
  kind  = "simple-state"
  label = "live\nreq_refs > 0"
}

node "cache-only" {
  kind  = "simple-state"
  label = "cache-only\nreq_refs = 0, cache_refs > 0"
}

node "pending" {
  kind  = "simple-state"
  label = "pending (slot, step)\nno refs; GPU may still read"
}

edge "start-free" {
  from = node.start
  to   = node.free
}

edge "alloc" {
  from  = node.free
  to    = node.live
  label = "alloc (from a reservation)"
}

edge "to-cache" {
  from  = node.live
  to    = node.cache-only
  label = "request ends, cache holds it"
}

edge "hit" {
  from  = node.cache-only
  to    = node.live
  label = "a cache hit refs it"
}

edge "release" {
  from  = node.live
  to    = node.pending
  label = "last ref dropped"
}

edge "evict" {
  from  = node.cache-only
  to    = node.pending
  label = "evicted"
}

edge "retire" {
  from  = node.pending
  to    = node.free
  label = "GPU event passed step"
}
