from difflib import get_close_matches
from typing import List


def get_closest_match(
    word: str,
    candidates: List[str],
    cutoff: float = 0.8,
) -> str:
    """Get the closest match to a word from candidates.

    Args:
        word: A string as input for matching against all candidates.
        candidates: A list of candidate strings to match against the
            input string, in which the closest match will be returned.
        cutoff: The similarity cutoff in the range of [0, 1].
            Candidates that don’t score at least that similar to
            word are ignored.

    Returns:
        The closest match to the `input` word in `candidates` with
        similarity of at least `cutoff`.

    Raises:
        ValueError: No match from `candidates` with similarity of
            at least `cutoff` to `word`.

    """
    _lower_case_to_original_candidate_dict = {
        __p.lower(): __p for __p in candidates
    }
    _lower_case_candidates = _lower_case_to_original_candidate_dict.keys()
    try:
        _lower_case_closest_match = get_close_matches(
            word=word.lower(),
            possibilities=_lower_case_candidates,
            n=1,
            cutoff=cutoff,
        )[0]
        _closest_match = _lower_case_to_original_candidate_dict[
            _lower_case_closest_match
        ]
        return _closest_match
    except IndexError:
        _error_msg = (
            f"Cannot find '{word}' in all candidates "
            f"({candidates}) within the similarity cutoff."
        )
        raise ValueError(_error_msg)
