"""The same text template is used for SFT data and web chat inference."""


def format_chat(messages):
    return '\n'.join(f'{"User" if m["role"] == "user" else "Assistant"}: {m["content"].strip()}'
                     for m in messages) + '\nAssistant:'
