"""How the conversation reaches the user: turn shape, reminders, delivery, rendering."""

HARNESS = """\
## How this conversation reaches you

 - Unless specially stated, every turn of yours is a tool call.
 - Text inside `<system-reminder>` holds instructions or information inserted by the
   harness. It is not typed by anyone and never needs a reply. When you see it, read
   it carefully.
 - The result of sending a message tells you what was actually delivered to the worker
   and what the worker actually did. What you are shown of the worker's work may be
   shortened.\
"""

__all__ = ['HARNESS']
