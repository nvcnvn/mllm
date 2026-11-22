// memory.h StepTables, as the kernels read it: [R][n_state][pos x R][n x R][R + 1 offsets][slots...][state x R]
#pragma once

struct Table {
  const device uint* t;
  int pos(int b) const { return t[2 + b]; }
  int n(int b) const { return t[2 + t[0] + b]; }
  uint slot(int b, int page) const { return t[3 + 3 * t[0] + t[2 + 2 * t[0] + b] + page]; }
  // Row b's state span (memory.mm state_offset): span byte x lives in arena byte state[x / unit] * unit + x % unit,
  // state = the row's n_state units after all pages.
  ulong state_byte(int b, int n_state, ulong unit, ulong x) const {
    uint s = t[3 + 3 * t[0] + t[2 + 3 * t[0]] + b * n_state + x / unit];
    return ulong(s) * unit + x % unit;
  }
};
