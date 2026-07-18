from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn


QUESTION_RE = re.compile(r"^Q\d+｜")
ANSWER_PREFIX = "参考回答："


def paragraph_text(paragraph_element) -> str:
    return "".join(
        text_element.text or ""
        for text_element in paragraph_element.iter(qn("w:t"))
    )


def text_node_payloads(body_element) -> list[str]:
    return [
        text_element.text or ""
        for paragraph_element in body_element.findall(qn("w:p"))
        for text_element in paragraph_element.iter(qn("w:t"))
    ]


def is_heading(paragraph_element) -> bool:
    properties = paragraph_element.find(qn("w:pPr"))
    if properties is None:
        return False
    style = properties.find(qn("w:pStyle"))
    if style is None:
        return False
    return style.get(qn("w:val"), "").startswith("Heading")


def should_stop(paragraph_element) -> bool:
    text = paragraph_text(paragraph_element)
    return not text or QUESTION_RE.match(text) is not None or is_heading(paragraph_element)


def merge_answers(source: Path, destination: Path) -> None:
    shutil.copyfile(source, destination)
    document = Document(str(destination))
    body = document._body._element
    before_payloads = text_node_payloads(body)

    answer_count = 0
    merged_paragraph_count = 0
    children = list(body)
    index = 0
    while index < len(children):
        element = children[index]
        if element.tag != qn("w:p") or not paragraph_text(element).startswith(ANSWER_PREFIX):
            index += 1
            continue

        answer_count += 1
        following_index = index + 1
        while following_index < len(children):
            following = children[following_index]
            if following.tag != qn("w:p") or should_stop(following):
                break

            for child in list(following):
                if child.tag != qn("w:pPr"):
                    element.append(child)
            body.remove(following)
            children.pop(following_index)
            merged_paragraph_count += 1

        index += 1

    document.save(str(destination))

    verified = Document(str(destination))
    verified_body = verified._body._element
    after_payloads = text_node_payloads(verified_body)
    remaining_answers = sum(
        1
        for p in verified_body.findall(qn("w:p"))
        if paragraph_text(p).startswith(ANSWER_PREFIX)
    )
    if after_payloads != before_payloads:
        mismatch_index = next(
            (
                index
                for index, (before_payload, after_payload) in enumerate(
                    zip(before_payloads, after_payloads)
                )
                if before_payload != after_payload
            ),
            min(len(before_payloads), len(after_payloads)),
        )
        print(f"before_node_count={len(before_payloads)}")
        print(f"after_node_count={len(after_payloads)}")
        print(f"mismatch_index={mismatch_index}")
        print(f"before_context={before_payloads[max(0, mismatch_index - 2):mismatch_index + 3]!r}")
        print(f"after_context={after_payloads[max(0, mismatch_index - 2):mismatch_index + 3]!r}")
        raise RuntimeError("Word text nodes changed during paragraph merge")
    if remaining_answers != answer_count:
        raise RuntimeError(
            f"Expected {answer_count} answer paragraphs after merge, found {remaining_answers}"
        )

    print(f"answers={answer_count}")
    print(f"merged_following_paragraphs={merged_paragraph_count}")
    print(f"output={destination}")
    print("visible_text_unchanged=true")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: merge_reference_answers.py SOURCE.docx OUTPUT.docx")
    merge_answers(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
