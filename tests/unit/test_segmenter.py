"""M3: Segmentierung des Verlaufs in logische Blöcke."""
from semanticswap.compression.segmenter import segment_messages
from semanticswap.memory.store import StoredMessage
from semanticswap.tokens import TokenCounter


def make_messages(n: int, chars: int = 200) -> list[StoredMessage]:
    return [
        StoredMessage(idx=i, role="user" if i % 2 == 0 else "assistant",
                      content=f"msg{i} " + "x" * chars, raw={})
        for i in range(n)
    ]


def test_messages_are_never_split():
    counter = TokenCounter(use_tiktoken=False)
    messages = make_messages(6)
    segments = segment_messages(messages, chunk_size=80, counter=counter)
    covered = []
    for seg in segments:
        covered.extend(range(seg.start_idx, seg.end_idx + 1))
    assert covered == list(range(6))  # lückenlos, keine Doppelungen


def test_small_history_is_one_segment():
    counter = TokenCounter(use_tiktoken=False)
    segments = segment_messages(make_messages(2, chars=20), 1000, counter)
    assert len(segments) == 1
    assert segments[0].start_idx == 0
    assert segments[0].end_idx == 1
    assert "USER:" in segments[0].text
    assert "ASSISTANT:" in segments[0].text


def test_chunk_size_forces_split():
    counter = TokenCounter(use_tiktoken=False)
    segments = segment_messages(make_messages(6, chars=200), 80, counter)
    assert len(segments) > 1
