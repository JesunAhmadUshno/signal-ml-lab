# PUBLICITY: signal-ml-lab (P17)

Publicity log under the documentation and publishing standard,
Section 7. Every finished level produces its written fan-out set.
Drafts are prepared by the loop. Actual social posting follows
the standing platform rules and gates. The loop itself never posts
to social.

Project state at this log: Level 1 Learn DONE, Level 2 Build DONE,
and Level 3 Deep DONE 2026-11-05. All measured results in this repo are on synthetic data
only. TensorFlow is not available in the build environment (import
fails with ModuleNotFoundError) and is not claimed anywhere in
this repo.

Fan-out set date: 2026-11-05. Project state: v1.0, Learn, Build,
and Deep done 2026-11-05, 43 tests passing. Deep detail: a stdlib http.server inference API (GET
/health, POST /predict, bad input 400) plus a CPU edge study (edge
is CPU by design, no GPU number claimed), float artifact accuracy
0.984 (492 of 500) at 3,530 bytes total (2,896 bytes weights), and
an int8 variant labelled a NumPy quantization study and not
TFLite at accuracy 0.984, 1,850 bytes, size ratio 0.6388. Timing
values in the results files vary between runs and are named as
such in every fan-out piece.


File map, 2026-11-08 (portfolio standard v2): the pieces below and in
fanout/ cite files that were folded. SPEC.md, NOVELTY.md and DOSSIER.md
now live in PROPOSAL.md; ROADMAP.md, docs/research-review.md and
docs/index.md live in REPORT.md and CHANGELOG.md. The dated text is
kept as written.

## Fan-out set (moved from README.md on 2026-11-08)

Written fan-out set for v1.0, dated 2026-11-05, in fanout/:

- fanout/blog-post.md: "The Published Method, Rebuilt", the full
  story with the main results claim gated by Jev claim_check
  (model jev-1.13.0, noul 0.86, route ACT, no block).
- fanout/tutorial.md: course module P17-signal-ml-lab. Every
  command in it was executed and passed on 2026-11-05 (all three
  runners, all per-level tests, full suite at 43 tests).
- fanout/website-page.md: project page copy only, not published.
- fanout/linkedin-post.md: DRAFT NOT POSTED, link as the
  first-comment line per the LinkedIn rule.
- fanout/paper-candidate.md: public replication brief at
  CANDIDATE. Submission is Jesun's tap, nothing submitted.
- fanout/release-notes.md: draft notes for the live releases
  v0.1, v0.5, and v1.0.

The loop never posts to social:
all social channels are held with named reasons in PUBLICITY.md.

## Channel log (2026-11-05, fan-out set)

| Channel | Outcome on 2026-11-05 |
| --- | --- |
| GitHub releases | Live: v0.1 https://github.com/JesunAhmadUshno/signal-ml-lab/releases/tag/v0.1 , v0.5 https://github.com/JesunAhmadUshno/signal-ml-lab/releases/tag/v0.5 , v1.0 https://github.com/JesunAhmadUshno/signal-ml-lab/releases/tag/v1.0 (verified with gh release list, 2026-11-05). Draft release notes for all three are in fanout/release-notes.md. No new release was created for the fan-out. |
| Portfolio index | Held: fan-out links are ready, the index update is a coordinator step outside this cycle. Named reason: held for index refresh. |
| Website page | Draft in fanout/website-page.md. Held: not published to jesunahmadushno.com. |
| Blog | Draft in fanout/blog-post.md. Held: not posted. |
| Tutorial | In repo: fanout/tutorial.md. Every command in it was run and verified on 2026-11-05 (three runners, three per-level test discoveries, scaffold smoke test, full suite at 43 tests). Results files whose only changes were timing fields were restored to the committed versions, so no result churn is in this change. |
| LinkedIn post | Draft in fanout/linkedin-post.md, marked DRAFT NOT POSTED, link carried as the first-comment line per the LinkedIn rule. Held: not posted, loop never posts. |
| X | Draft held: no X copy is shipped in this set beyond the LinkedIn draft source text. Held under the standing social rules; posting is outside the loop. |
| Facebook | Draft held: no Facebook copy posted. Standing rule applies (no link in body or comments, keyword call to action fulfilled by DM). Held: not posted, loop never posts. |
| Instagram | Draft held: no Instagram copy posted. Standing rule applies (no clickable link in caption, link in bio). Held: not posted, loop never posts. |
| Threads | Draft held: no Threads copy posted. Threads waits for Jesun's tap per post, hash-locked on exact text and media. Held: awaiting Jesun's tap. |
| Video | Held: engine queue, January production for lab projects per the December Compression Protocol. No video was produced in this cycle. |
| Book chapter | Draft angle in fanout/blog-post.md and SPEC.md ("The Published Method, Rebuilt"). Held: chapter draft follows the finished project set; not a published chapter. |
| Paper candidate | Draft brief in fanout/paper-candidate.md (public replication of the published method). Submission is Jesun's tap. Nothing submitted, no venue contacted. The publications tracker is updated by the coordinator, not by this cycle. |

## Jev gate (2026-11-05)

claim_check on the blog main results paragraph, extracted
byte-exact from fanout/blog-post.md by script. Spec:
hidden_files/jev-p17-fanout-claim-spec-2026-11-05.json in the goal
workspace. Results:
hidden_files/jev-p17-fanout-claim-results-2026-11-05.json. Model
jev-1.13.0, question id supported, noul 0.86, route ACT (clear at
0.85 or above, block at 0.10 or below, review between), input
tokens 1738, output tokens 20. Arithmetic was pre-computed in the
spec evidence (492 of 500 is 0.984, 1850 of 2896 is 0.6388, 2400
plus 500 is 2900, test totals 14 plus 20 is 34 and 34 plus 9 is
43). No claim was blocked, and no second gate run was needed.

## Actions polish pack (2026-11-05)

Applied through a pull request on 2026-11-05, following the Actions
polish pack standard. The pack adds: CI on a Python 3.11 and 3.12
matrix with a ruff lint step (all pre-existing P17 CI steps
preserved: scaffold smoke test, install numpy, Learn tests and run,
Build tests and run), CodeQL scanning, Dependabot for pip and
GitHub Actions, a release workflow that attaches a source tarball
named signal-ml-lab-<tag>.tar.gz to v* tag releases, Labeler,
Greetings (snake_case inputs for actions/first-interaction@v3:
repo_token, issue_message, pr_message), Stale in comment-only mode
(days-before-close: -1, never closes), and a docs summary page at
docs/index.md for GitHub Pages. Ruff scope is recorded in
ruff.toml (E, F, I, line-length 120). Ruff fixes in this pull
request were style only: 74 initial errors, fixed by ruff fix
output (9), manual splitting of multiple statements on one line,
import ordering, and wrapping of long lines, including string
literals split by concatenation with identical values, with no
change to behaviour. requirements.txt records the NumPy dependency
honestly: this repo uses NumPy, it is not standard-library-only,
and TensorFlow is not a dependency and is not claimed.

| Channel | Outcome on 2026-11-05 |
| --- | --- |
| GitHub Pages | Docs page added at docs/index.md in this pull request. Pages enablement and live build status are verified after merge and recorded in the batch ledger; no live claim is made in this file before that check. |
| PyPI publishing | Held: needs a PyPI token, a new credential. Jesun's tap only. Not set up. |
| Social channels (LinkedIn, X, Facebook, Instagram, Threads) | Held: no posts made. The loop never posts to social; Threads waits for Jesun's tap per post under the standing gates. |

