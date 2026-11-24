# Requests own lists of slot numbers; the CPU copies them into a step table; the GPU kernel turns them into addresses.

node "req-a" {
  kind  = "table"
  label = "Request A"
  row "pages" {
    type = "0, 3, 6, …"
  }
  row "state" {
    type = "1, 4, … (50 units)"
  }
}

node "req-b" {
  kind  = "table"
  label = "Request B"
  row "pages" {
    type = "2, 9, …"
  }
  row "state" {
    type = "5, … (50 units)"
  }
}

node "table" {
  kind  = "table"
  label = "Step table"
  row "rows" {
    type = "R"
  }
  row "pos" {
    type = "first position, per row"
  }
  row "n" {
    type = "tokens this step, per row"
  }
  row "offsets" {
    type = "where each row's slots start"
  }
  row "slots" {
    type = "page slots, then state slots"
  }
}

node "kernel" {
  kind  = "service"
  label = "GPU kernel\nslot × unit + offset"
}

node "arena" {
  kind  = "storage"
  label = "arena: N equal units\nnever move"
}

edge "a-table" {  # the CPU copies both lists in, every step
  from  = node.req-a
  to    = node.table.row.slots
}

edge "b-table" {
  from = node.req-b
  to   = node.table.row.slots
}

edge "table-kernel" {
  from  = node.table
  to    = node.kernel
  label = "GPU reads"
}

edge "kernel-arena" {
  from  = node.kernel
  to    = node.arena
}
