"""Front relay eval fixtures: what the voice front says and hands to Claude in the situations that went wrong live
(2026-10-03, Qwen3-4B front) or in earlier eval runs (Qwen3-4B, Qwen3.5-9B, Qwen3.6-35B-A3B).

Each fixture is the conversation so far, then the turn under test:
  prior          front turns as (user words, delegate, say), and announcements already heard as ("report", text, say)
  waiting        results that arrived while the user was talking (FrontSession gives them to the user's turn)
  user           the user's words for this turn, or None for an announcement turn
  reports        for an announcement turn, the results announced together
  question_open  the last announced report asked the user something
  sent           delegations handed off in the last few minutes, as (said, request), for the repeat guard
  criteria       the pass criterion, in words
  rubric         keep: one regex must match the speech; also: every one must; forbid: none may
                 first_keep: one must match the first sentence; max_sentences
                 delegate: "yes", "no" or "any"; deleg_keep / deleg_forbid: over the delegated request
                 said_verbatim: a delegation carries the user's words unchanged
                 note: the emitted note has these fields; answer_or_delegation: a note with answer=true, or a
                 delegation carrying the user's words verbatim; note_record: record the note kind, unscored
  extra          checks added after a fixture's first results, scored apart so its own score stays comparable

Global checks on every fixture:
  INVENTED  a user turn that hands nothing off announces no action the front cannot take
  DOUBLED   no turn says "asked" twice
"""

INVENTED = [r"\bI(?:'ll| will| am going to|'m going to) (?!be\b|report\b|let you\b|tell you\b|pass\b|share\b)\w+",
            r"\blet me (?!know\b)\w+"]  # "I'll let you know / pass it on / report" are kept promises
DOUBLED = r"\basked\b.*\basked\b"
GUESS = [r"\blikely\b", r"\bprobably\b", r"\bdue to\b", r"\bbecause\b", r"\bso the\b", r"\bseems? to\b",
         r"\bmust have\b"]

R1 = ("I don't have results yet. The `sidecar-log` worker is reading the live-vibe logs in the background: the sidecar "
      "log and the front-server log from this session's start, plus the Windows-side player log if there is one. "
      "It's only reading; it won't start, stop or kill anything or use the GPU.\n\nIt will report which speaker output "
      "was actually picked, whether speech-to-text and Kokoro landed on CUDA, whether the voice front model loaded, "
      "and every warning or error with its timestamp. I'll pass on the findings when it finishes.")
R1_SAY = ("The sidecar-log worker is currently reading the live-vibe logs in the background. I'll share the findings "
          "when it finishes.")
R2 = ("The logs from this session show no errors and no warnings that affect anything. I checked the front-server log "
      "myself; the sidecar summary comes from the worker's read of its log.\n\n"
      "- **Speaker:** the Windows player was picked and is running on Speakers (Stealth 700X Gen 3) through WASAPI.\n"
      "- **GPU:** speech-to-text, Kokoro and the echo canceller all came up on the GPU.\n\n"
      "This only covers the first minute or so of the session, and it says nothing about how the audio sounds. "
      "Whether the static is gone still needs your ears. Nothing is running in the background now.")
R2_SAY = "The logs show no errors. Whether the static is gone still needs your ears."
R4 = ("One change in the code targets the static: the Windows-side player in 0.5.0 (`8cc0216`). Today's session log "
      "shows it's in use. Nobody has confirmed by ear that the static is gone.\n\n"
      "| Change | Version | What it did |\n|---|---|---|\n| Windows-side player | 0.5.0 | Bypasses WSLg audio |\n\n"
      "**Still open:** whether the static is gone, and whether echo cancellation holds up on real speakers. Both "
      "need you to listen.")
R5 = ("The Windows player is now the default speaker backend in 0.5.1, and all 41 session checks pass. Nothing is "
      "pushed yet; it waits for your OK.")
R6 = ("The `fix/live-vibe-native-rate` branch is local only and made no audible difference. Shall I delete the "
      "branch?")
START = ("All right, let's try this again.", "Check the live-vibe logs for errors or warnings.", "Alright, let's go.")

F = [
    # -- the first five (rubrics as first scored) --------------------------------------------------------------
    dict(
        id="A_what_fix",
        criteria="'so what was the fix?' while the worker's result waits: no stale 'still running', no guessed cause",
        prior=[START, ("report", R1, R1_SAY)], waiting=[R2],
        user="Wow, so now it sounds perfect. So what was the fix?",
        rubric=dict(keep=[r"\bno errors\b", r"\bwindows\b", r"\bplayer\b", r"\basked\b", r"\bchecking\b", r"\bon it\b"],
                    forbid=[r"still (running|reading|working)", r"hasn't finished", r"not finished",
                            r"\bcurrently reading"] + GUESS[:3] + [r"\bwas (in|resolved|adjusted)\b", r"\bno fix\b"],
                    delegate="any"),
        extra=dict(deleg_keep=[r"\bfix", r"\bstatic\b"], deleg_forbid=[r"\bwait for\b"],
                   criteria="a delegation, if any, names the fix or the static (not 'wait for the worker')"),
    ),
    dict(
        id="B_confirm",
        criteria="a confirmation ('it sounds perfect, it is fixed'): a short reply, no delegation",
        prior=[START, ("report", R1, R1_SAY), ("report", R2, R2_SAY)], waiting=[],
        user="Yeah, well, yeah, it sounds perfect. There is no static. It is fixed.",
        rubric=dict(keep=[r"\b(great|glad|good|got it|noted|thanks|perfect|static)\b"],
                    forbid=[r"\b(I'll|I will) (look|check|investigate|run)", r"still (running|reading)"],
                    delegate="no"),
    ),
    dict(
        id="B_followup",
        criteria="'did you get that?': yes, no invented fix",
        prior=[START, ("report", R2, R2_SAY),
               ("Yeah, it sounds perfect. There is no static. It is fixed.", "", "Got it, the static is gone.")],
        waiting=[], user="Yeah, so did you get that? Because you haven't really followed up.",
        rubric=dict(keep=[r"\b(yes|got it|passed|noted|told)\b"],
                    forbid=[r"still (running|reading)", r"\bfixed it\b", r"\bI fixed\b"], delegate="any"),
        extra=dict(first_keep=[r"^\W*(yes|yeah|yep|got it|noted|understood|i did|i got)\b"],
                   criteria="the first sentence is a direct yes (yes / got it / noted / understood)"),
    ),
    dict(
        id="C_stale_status",
        criteria="'is it done yet?' with a review handed off and no report since: says it asked, never a status. The "
                 "log worker's report is in the history, so asking for it again is a re-ask",
        prior=[START, ("report", R1, R1_SAY), ("report", R2, R2_SAY),
               ("Okay, so please explain to me how you fixed it.",
                "Review the code changes from the session and explain how the static was fixed.", "On it.")],
        waiting=[], user="So, is it done yet?",
        rubric=dict(keep=[r"\basked\b", r"\bwaiting\b", r"\bnot yet\b", r"\bno (word|report|answer)\b",
                          r"\bhaven't heard\b", r"\bwhen it\b", r"\bchecking\b"],
                    forbid=[r"still (running|reading|working)", r"\b(it's|it is) (done|finished|complete)",
                            r"\bhas finished\b", r"\bnot finished\b", r"\bhasn't finished\b"],
                    delegate="any"),
        extra=dict(deleg_keep=[r"\bdone\b", r"\bstatus\b", r"\bfinish", r"\bcomplete"],
                   deleg_forbid=[r"\berrors or warnings\b"],
                   criteria="a delegation, if any, asks whether the pending review is done, and does not re-ask the "
                            "finished log report ('errors or warnings')"),
    ),
    dict(
        id="E_combined",
        criteria="two results announced together: both retold (the logs, and the Windows player change), no claim "
                 "that the static is fixed",
        prior=[START], waiting=[], user=None, reports=[R2, R4],
        rubric=dict(keep=[r"\b(windows|player|8cc0216)\b"], also=[r"\b(no errors|logs|warnings)\b"],
                    forbid=[r"\bstill running\b", r"\b(it's|it is|everything is|static is) (fixed|done)\b"],
                    delegate="no"),
    ),
    # -- added from the 4B, 9B and 35B runs ------------------------------------------------------------------------
    dict(
        id="N1_invented",
        criteria="a confirmation after a finished result: a short reply, no delegation, no announced action the front "
                 "cannot take (mark verified, stop tracking, update status)",
        prior=[START, ("report", R2, R2_SAY)], waiting=[],
        user="Great, that works. You can consider it verified.",
        rubric=dict(keep=[r"."], forbid=[r"\bmark", r"\btrack", r"\bupdate", r"\bstatus\b", r"\bverif(y|ied) it\b"],
                    delegate="no", max_sentences=2),
    ),
    dict(
        id="N2_single_report",
        criteria="one report: name its specific (the Windows player or 0.5.1), keep 'not pushed', infer no cause",
        prior=[START], waiting=[], user=None, reports=[R5],
        rubric=dict(keep=[r"\bwindows player\b", r"\b0\.5\.1\b"],
                    also=[r"\bnot (been )?pushed|\bnothing is pushed|\bwaits? for your ok|\byour (ok|approval)"],
                    forbid=GUESS + [r"(?<!not )(?<!nothing is )\bpushed\b(?! yet)", r"\breleased\b"], delegate="no"),
    ),
    dict(
        id="N6_answer",
        criteria="'yes, delete it' after a report that asked 'Shall I delete the branch?': Claude gets the answer (a "
                 "note with answer=true, or a delegation carrying the user's words verbatim), and the front claims "
                 "nothing was deleted yet",
        prior=[START, ("report", R6, "The native-rate branch is local only and made no difference. Shall I delete "
                                     "the branch?")],
        question_open=True, waiting=[], user="Yes, delete it.",
        rubric=dict(keep=[r"."], forbid=[r"\b(deleted|removed|done)\b", r"\b(it's|it is|branch is) gone\b"],
                    delegate="any", answer_or_delegation=True),
    ),
    dict(
        id="N7_smalltalk",
        criteria="'how are you?' with nothing pending: a short direct answer (2 sentences at most), no delegation; "
                 "the note kind is recorded (ask costs a Claude turn)",
        prior=[("Hi there.", "", "Hi! What can I do for you?")], waiting=[], user="How are you?",
        rubric=dict(keep=[r"\b(good|well|great|fine|doing)\b"], forbid=[], delegate="no", max_sentences=2,
                    note_record=True),
    ),
    dict(
        id="N8_correction",
        criteria="'no, I meant the Windows side, not WSL' right after a delegation: delegate again with the correction "
                 "(not blocked as a repeat), carrying the user's words verbatim",
        prior=[("Make the audio play on the WSL side.", "Make the sidecar play its audio through WSL.", "On it.")],
        sent=[("Make the audio play on the WSL side.", "Make the sidecar play its audio through WSL.")],
        waiting=[], user="No, I meant the Windows side, not WSL.",
        rubric=dict(keep=[r"."], forbid=[], delegate="yes", deleg_keep=[r"\bwindows\b"], said_verbatim=True),
    ),
    dict(
        id="N9_race_unrelated",
        criteria="a result waits while the user asks something unrelated ('what time is it in Tokyo?'): no time made "
                 "up, delegate or answer briefly, and the waiting report's status is still retold (in the reply or a "
                 "following announcement) without invention",
        prior=[START, ("report", R1, R1_SAY)], waiting=[R2], user="What time is it in Tokyo right now?",
        rubric=dict(keep=[r"\b(no errors|logs?|warnings|static|ears)\b"],
                    forbid=GUESS + [r"\bstill (running|reading)\b", r"\b\d{1,2}(:\d\d)? ?(am|pm|a\.m\.|p\.m\.)"],
                    delegate="any"),
    ),
    dict(
        id="N10_goodbye",
        criteria="'ok thanks, bye': a short goodbye, no delegation",
        prior=[START, ("report", R2, R2_SAY)], waiting=[], user="Ok thanks, bye.",
        rubric=dict(keep=[r"\b(bye|goodbye|see you|take care)\b"], forbid=[], delegate="no", max_sentences=2),
    ),
]
