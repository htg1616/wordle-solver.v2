# Wordle Project Starter Code

This directory contains a local grader and a minimal example solver.

Use these files to check that your solver follows the required HTTP protocol and
can solve small local problem instances.

## Files

- `grader.py`: local grader for testing your solver.
- `team00.py`: minimal example solver/server.
- `team00_problem.json`: example problem instance.

You should mainly edit `team00.py` or make your own team file based on it.

## Setup

The local grader uses the `requests` package.

```bash
pip install requests
```

## What You Need To Implement

Your solver must run as an HTTP server. It must:

- read the port number from the `PORT` environment variable;
- handle `POST /start_problem`;
- handle `POST /act`;
- remember its own guesses and feedback history;
- return only candidate words provided by the grader;
- decide when to `guess` and when to `submit`.

The starter `team00.py` already implements the server wrapper. Its solving logic
is intentionally simple, so you should replace it with your own method.

## Feedback Format

The grader does not send raw numeric feedback such as `01020`.

Instead, it sends a deterministic English string with one line per guessed
letter. For example:

```text
The first letter 'm' is absent.
The second letter 'e' is misplaced.
The third letter 'l' is absent.
The fourth letter 'e' is correct.
The fifth letter 'e' is absent.
```

The meanings are:

- `absent`: feedback digit `0`;
- `misplaced`: feedback digit `1`;
- `correct`: feedback digit `2`.

This format is deterministic, so you can parse it with ordinary string
processing. No LLM is needed.

## HTTP Protocol

### `POST /start_problem`

The grader sends the candidate words and known noise settings:

```json
{
  "problem_id": "1",
  "candidate_words": ["hello", "world"],
  "noise_probability": 0.33,
  "two_letter_noise_probability": 0.33,
  "force_noise_turns": 3,
  "max_turns": 100,
  "guess_time_budget": 60
}
```

Your server should initialize its state and respond with:

```json
{}
```

### `POST /act`

The grader sends only the most recent feedback:

```json
{
  "problem_id": "1",
  "turn": 1,
  "feedback": null
}
```

On the first turn, `feedback` is `null`. On later turns, it is the feedback for
your previous guess.

Your server must respond with either a guess:

```json
{
  "action": "guess",
  "word": "hello"
}
```

or a final submission:

```json
{
  "action": "submit",
  "word": "hello"
}
```

The `word` must be one of the candidate words.

## Noise Model

For each guess, feedback is computed against an effective secret:

- probability `1 - p - q`: the true secret;
- probability `p`: the true secret with one letter changed;
- probability `q`: the true secret with two letters changed.

The perturbed effective secret may be a non-word.

The first `force_noise_turns` turns are guaranteed to contain at least one noisy
feedback event. With the default settings, this means at least one noisy event
within the first three guess turns.

## Timing

Default local grader limits:

- `--startup-delay 5`: grader waits 5 seconds after starting your server;
- `--start-timeout 5`: `/start_problem` must respond within 5 seconds;
- `--guess-time-budget 60`: all `/act` calls for one problem share 60 seconds.

The grader starts a fresh solver process for each problem.

## Problem File

The default problem file is `team00_problem.json`:

```json
{
  "secret_word": "hello",
  "candidate_words": ["hello", "world"]
}
```

The local grader also accepts a list of problems:

```json
[
  {
    "secret_word": "hello",
    "candidate_words": ["hello", "world"]
  },
  {
    "secret_word": "flame",
    "candidate_words": ["flame", "frame", "crane"]
  }
]
```

You do not need to provide `problem_id`; the local grader will assign one.

## Running The Local Grader

From this directory:

```bash
python grader.py
```

Run a different solver:

```bash
python grader.py --solver-script my_team.py
```

Run a different problem file:

```bash
python grader.py --problems-path my_problems.json
```

Change the noise settings:

```bash
python grader.py \
  --noise-probability 0.33 \
  --two-letter-noise-probability 0.33
```

The grader prints a JSON summary containing the status, score turns, and
feedback history for each local problem.
