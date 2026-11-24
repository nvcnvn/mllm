# M4 Pro Mac mini: three processors, one memory pool. Bandwidths and TFLOPS measured on this machine.

node "cpu" {
  kind  = "service"
  label = "CPU, 14 cores\nbookkeeping, tokenizer"
}

node "gpu" {
  kind  = "service"
  label = "GPU, 7.5 TFLOPS\ndecode, attention, sampling\nprefill by default"
}

node "ane" {
  kind  = "service"
  label = "Neural Engine, ~13 TFLOPS\nprefill dense blocks only"
}

node "ram" {
  kind  = "database"
  label = "LPDDR5X, 64 GB, one pool\n273 GB/s peak"
}

edge "cpu-ram" {
  from  = node.cpu
  to    = node.ram
  label = "counters, step tables"
}

edge "gpu-ram" {
  from  = node.gpu
  to    = node.ram
  label = "210–238 GB/s"
}

edge "ane-ram" {
  from  = node.ane
  to    = node.ram
  label = "~39 GB/s weights"
}

edge "ane-gpu" {
  from  = node.ane
  to    = node.gpu
  kind  = "async"
  label = "IOSurface, zero-copy"
}
