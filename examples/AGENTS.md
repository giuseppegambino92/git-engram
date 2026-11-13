## Repository memory (git-engram)

- Before editing any file, call `engram_preflight_check(file_path, symbol_name, intended_change_query)` and obey
  `<invariants>`, `<load_bearing_code>` and `<anti_context_do_not_repeat>`. Check every file listed in
  `<co_change_coupling>` for a matching change.
- Run tests through `git-engram exec -- <test command>` so failed attempts are remembered automatically.
- If you abandon an approach, call `engram_record_failure` with the diff and the error.
- When you discover a rule that must always hold, call `engram_record_invariant`.
