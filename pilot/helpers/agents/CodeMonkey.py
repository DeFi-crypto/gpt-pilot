import os.path
import re

from helpers.AgentConvo import AgentConvo
from helpers.Agent import Agent
from helpers.files import get_file_contents
from logger.logger import logger


class CodeMonkey(Agent):
    save_dev_steps = True

    def __init__(self, project, developer):
        super().__init__('code_monkey', project)
        self.developer = developer

    def implement_code_changes(self, convo, task_description, code_changes_description, step, step_index=0):
        convo = AgentConvo(self)

        files = self.project.get_all_coded_files()

        if 'path' not in step or 'name' not in step:
            files_to_change = self.identify_files_to_change(convo, code_changes_description)
            assert len(files_to_change) == 1  # FIXME we should loop over this
            step['path'] = os.path.dirname(files_to_change[0])
            step['name'] = os.path.basename(files_to_change[0])

        rel_path, abs_path = self.project.get_full_file_path(step['path'], step['name'])

        for f in files:
            if (f['path'] == step['path'] or (os.path.sep + f['path'] == step['path'])) and f['name'] == step['name'] and f['content']:
                file_content = f['content']
                break
        else:
            # If we didn't have the match (because of incorrect or double use of path separators or similar), fallback to directly loading the file
            file_content = get_file_contents(abs_path, self.project.root_path)['content']
            if isinstance(file_content, bytes):
                file_content = "... <binary file, content omitted> ..."

        file_name = os.path.join(rel_path, step['name'])

        llm_response = convo.send_message('development/implement_changes.prompt', {
            "code_changes_description": code_changes_description,
            "file_content": file_content,
            "file_name": file_name,
            "directory_tree": self.project.get_directory_tree(True),
        })

        exchanged_messages = 2

        if "reports.js" in llm_response and "// Insert updated script tag content as provided" in llm_response:
            print("HERE")

        for retry in range(5):
            # Modify a copy of the content in case we need to retry
            content = file_content

            # Split the response into pairs of old and new code blocks
            self.pattern = re.compile(r"```([a-z0-9]+)?\n(.*?)\n```\s*", re.DOTALL)
            blocks = []
            for block in self.pattern.findall(llm_response):
                blocks.append(block[1])

            if len(blocks) == 0:
                print(f"No changes required for {step['name']}")
                break

            if len(blocks) % 2 != 0:
                llm_response = convo.send_message('utils/llm_response_error.prompt', {
                    "error": "Each change should contain old and new code blocks."
                })
                exchanged_messages += 2
                continue

            # Replace old code blocks with new code blocks
            try:
                for old_code, new_code in zip(blocks[::2], blocks[1::2]):
                    content = self.replace(content, old_code, new_code)
                # Success, we're done with the file
                break
            except ValueError as err:
                # we can't match old code block to the original file, retry
                llm_response = convo.send_message('utils/llm_response_error.prompt', {
                    "error": str(err),
                })
                exchanged_messages += 2
                continue
        else:
            logger.warning(f"Unable to implement code changes after 5 retries: {code_changes_description}")

        if content and content != file_content:
            self.project.save_file({
                'path': step['path'],
                'name': step['name'],
                'content': content,
            })

        convo.remove_last_x_messages(exchanged_messages)
        return convo

    def identify_files_to_change(self, convo, code_changes_description):
        convo = AgentConvo(self)
        llm_response = convo.send_message('development/identify_files_to_change.prompt', {
            "code_changes_description": code_changes_description,
            "files": self.project.get_all_coded_files(),
        })

        lines = llm_response.splitlines()
        files = [ line.strip() for line in lines if '```' not in line ]
        return files

    @staticmethod
    def replace(haystack: str, needle: str, replacement: str) -> str:
        """
        Replace `needle` text in `haystack`, allowing that `needle` is not
        indented the same as the matching part of `haystack` and
        compensating for it.

        Example:
        >>> haystack = "def foo():\n    pass"
        >>> needle = "pass"
        >>> replacement = "return 42"
        >>> replace(haystack, needle, replacement)
        "def foo():\n    return 42"

        If `needle` is not found in `haystack` even with indent compensation,
        return `None`.
        """

        def indent_text(text: str, indent: int) -> str:
            return "\n".join((" " * indent + line) for line in text.splitlines())

        # Try from the largest indents to the smalles so that we know the correct indentation of
        # single-line old blocks that would otherwise match with 0 indent as well. If these single-line
        # old blocks were then replaced with multi-line blocks and indentation wasn't not correctly re-applied,
        # the new multiline block would only have the first line correctly indented. We want to avoid that.
        for indent in range(128, -1, -1):
            text = indent_text(needle, indent)
            if text not in haystack:
                # If there are empty lines in the old code, `indent_text` will indent them as well. The original
                # file might not have them indented as they're empty, so it is useful to try without indenting
                # those empty lines.
                text = "\n".join(
                    (line if line.strip() else "")
                    for line
                    in text.splitlines()
                )
            if text in haystack:
                if haystack.count(text) > 1:
                    raise ValueError(
                        f"Old code block found more than once in the original file:\n```\n{needle}\n```\n\n"
                        "Please provide larger blocks (more context) to uniquely identify the code that needs to be changed."
                    )
                indented_replacement = indent_text(replacement, indent)
                return haystack.replace(text, indented_replacement)

        raise ValueError(
            f"Old code block not found in the original file:\n```\n{needle}\n```\n"
            "Old block *MUST* contain the exact same text (including indentation, empty lines, etc.) as the original file "
            "in order to match."
        )