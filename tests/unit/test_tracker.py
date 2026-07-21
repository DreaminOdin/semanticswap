"""M2: Session-Wiedererkennung per Prefix-Hash-Kette (ADR-003)."""
from semanticswap.memory.store import Store
from semanticswap.session.tracker import SessionTracker, chain_hashes, message_hash


def msgs(*contents, system="You are helpful."):
    out = [{"role": "system", "content": system}]
    role = "user"
    for c in contents:
        out.append({"role": role, "content": c})
        role = "assistant" if role == "user" else "user"
    return out


def test_message_hash_normalizes_whitespace():
    a = {"role": "user", "content": "Hallo   Welt\n"}
    b = {"role": "user", "content": "Hallo Welt"}
    assert message_hash(a) == message_hash(b)


def test_chain_depends_on_prefix():
    c1 = chain_hashes(msgs("a", "b"))
    c2 = chain_hashes(msgs("a", "x"))
    assert c1[0] == c2[0]  # system identisch
    assert c1[1] == c2[1]  # "a" identisch
    assert c1[2] != c2[2]  # divergiert ab "b"/"x"


def test_new_session_created_without_match():
    tracker = SessionTracker(Store(":memory:"))
    res = tracker.resolve(msgs("hallo"))
    assert res.is_new
    assert res.matched_count == 0


def test_strict_append_reuses_session():
    store = Store(":memory:")
    tracker = SessionTracker(store)
    m1 = msgs("frage 1")
    res1 = tracker.resolve(m1)
    full1 = m1 + [{"role": "assistant", "content": "antwort 1"}]
    tracker.record(res1.session_id, full1)

    m2 = full1 + [{"role": "user", "content": "frage 2"}]
    res2 = tracker.resolve(m2)
    assert res2.session_id == res1.session_id
    assert not res2.is_new
    assert res2.matched_count == 3  # system + frage + antwort


def test_edit_forks_session():
    store = Store(":memory:")
    tracker = SessionTracker(store)
    m1 = msgs("frage 1")
    res1 = tracker.resolve(m1)
    tracker.record(res1.session_id, m1 + [{"role": "assistant", "content": "antwort 1"}])

    edited = msgs("frage 1 - EDITIERT")  # gleiche system msg, andere Frage
    res2 = tracker.resolve(edited)
    assert res2.session_id != res1.session_id
    assert res2.is_new
    assert res2.forked_from == res1.session_id
    assert res2.matched_count == 1  # nur der System-Prompt matcht


def test_fork_inherits_archive_if_before_divergence():
    store = Store(":memory:")
    tracker = SessionTracker(store)
    m1 = msgs("frage 1")
    res1 = tracker.resolve(m1)
    tracker.record(res1.session_id, m1 + [{"role": "assistant", "content": "antwort 1"}])
    store.set_archive(res1.session_id, 1, "ARCHIV-PROMPT")

    edited = msgs("frage 1 - EDITIERT")
    res2 = tracker.resolve(edited)
    fork = store.get_session(res2.session_id)
    assert fork.archived_upto == 1
    assert fork.archive_prompt == "ARCHIV-PROMPT"


def test_fork_shares_memory_space():
    """ADR-009: Forks erben die memory_id und sehen damit Segmente/Embeddings
    der Ursprungs-Session."""
    store = Store(":memory:")
    tracker = SessionTracker(store)
    m1 = msgs("frage 1")
    res1 = tracker.resolve(m1)
    tracker.record(res1.session_id, m1 + [{"role": "assistant", "content": "antwort 1"}])

    res2 = tracker.resolve(msgs("frage 1 - EDITIERT"))
    fork = store.get_session(res2.session_id)
    original = store.get_session(res1.session_id)
    assert fork.memory_id == original.memory_id == res1.session_id


def test_explicit_session_id_takes_precedence():
    store = Store(":memory:")
    tracker = SessionTracker(store)
    res = tracker.resolve(msgs("hallo"), explicit_id="meine-session")
    assert res.session_id == "meine-session"
    tracker.record("meine-session", msgs("hallo"))
    res2 = tracker.resolve(msgs("hallo"), explicit_id="meine-session")
    assert res2.session_id == "meine-session"
    assert res2.matched_count == 2
