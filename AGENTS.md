# Workflow Constraints
- Always read in all the `specs/*goal*` files before making any changes.
- Always run `cargo clippy` and handle/fix all the issues it highlights.
- Always format Rust files before being finished with edits (this can happen as the last step).
  Use the `cargo format` alias (`cargo fmt --all`) to format the whole workspace. When
  formatting individual files with `rustfmt` directly, always pass `--edition 2024` — omitting
  it defaults to an older edition and will error on this codebase.

# General Constraints
- Non-code file formatting, including documentation and this rules file, must be hard-wrapped to a
  maximum of 100 characters per line wherever practical. Rust files are formatted by `rustfmt`.

# Tech Debt & Design Decisions
- Avoid exacerbating tracked tech debt (`specs/tech_debt.md`). Recommend the correct fix
  over temporary patches that would be undone during debt remediation. If the fix is
  substantially costlier, escalate the decision to a human reviewer rather than choosing
  the path that worsens debt.

# Unit Testing Standards
- All unit tests must be granular and test a single, isolated concept or behavior.
- Do NOT write monolithic unit tests that combine multiple assertions or steps representing different
  behaviors (e.g., do not combine testing format support and I/O error handling into one test).
  (Integration and end-to-end tests are exempt from this strict single-behavior rule).
- Use descriptive test names that clearly convey the specific behavior being tested (e.g.,
  `test_save_unsupported_extension` rather than `test_save_errors`).
- Share setup code using helper functions (e.g., `create_test_image()`) rather than repeating
  setup logic across multiple tests.
- Adhere strictly to the "Testing Pyramid" philosophy: Maximize heavily isolated, fast-executing
  unit tests. Minimize slow, brittle integration and end-to-end (e2e) tests. Use unit tests strictly
  for permutations and edge-case validation, reserving e2e tests only for integration "glue" 
  verification (e.g., passing data across boundaries).

# Documentation & Tone
- **Objective but Explanatory:** Technical documentation should be readable and provide necessary
  context, not just a terse list of commands. It is good to explain *why* a command is run or *how*
  a system works to aid understanding. However, keep the language objective. Avoid flowery,
  self-aggrandizing, or sales-like adjectives (e.g., "massive," "flawless," "heavily leveraging")
  that do not add to technical clarity. Present mechanics as neutral architectural details.
