"""Held-out website contract probes. No tool execution or external side effects."""
import json
from pathlib import Path
from prepare_data import ROOT, digest, render_samples, template_renderer, validate_calls
from tokenizers import Tokenizer

def main():
    tools = json.loads((ROOT/'website-tools.json').read_text())
    # Prompts are excluded from the synthetic training curriculum.
    cases = [
        ('What is 87 times 96? Use the calculator.', 'calculator', {'expression':'87 * 96'}),
        ('Calculate the expression (240 - 35) / 5.', 'calculator', {'expression':'(240 - 35) / 5'}),
        ('Find the current time in Europe/Berlin.', 'get_current_time', {'timezone':'Europe/Berlin'}),
        ('What day and time is it where I am?', 'get_current_time', {}),
        ('Show all my saved notes.', 'manage_notes', {'action':'list'}),
        ('Find notes containing the phrase annual inspection.', 'manage_notes', {'action':'search','query':'annual inspection'}),
        ('Save a note titled "Call supplier" containing "Ask about delivery on Tuesday".', 'manage_notes', {'action':'create','title':'Call supplier','content':'Ask about delivery on Tuesday'}),
        ('Delete the note with ID note-8472.', 'manage_notes', {'action':'delete','id':'note-8472'}),
        ('Change the content of note ID note-916 to "Appointment at noon".', 'manage_notes', {'action':'update','id':'note-916','content':'Appointment at noon'}),
        ('Execute this Python exactly: print(sum([9, 14, 28]))', 'run_code', {'language':'python','code':'print(sum([9, 14, 28]))'}),
        ('Run this JavaScript exactly: console.log([8, 3, 6].sort((a,b) => a-b))', 'run_code', {'language':'javascript','code':'console.log([8, 3, 6].sort((a,b) => a-b))'}),
        ('Search the web using the query "Unsloth Qwen3.5 tool calling".', 'web_search', {'query':'Unsloth Qwen3.5 tool calling'}),
        ('Search online for "New York weather tomorrow".', 'web_search', {'query':'New York weather tomorrow'}),
        ('Read https://docs.ollama.com/capabilities/tool-calling', 'fetch_webpage', {'url':'https://docs.ollama.com/capabilities/tool-calling'}),
        ('Fetch https://example.org/article and limit the content to 1200 characters.', 'fetch_webpage', {'url':'https://example.org/article','max_length':1200}),
        ('Good morning, how are you?', None, 'I am ready to help.'),
        ('Translate "good night" into French.', None, 'Bonne nuit.'),
        ('Make "Give me the report" more polite.', None, 'Could you please send me the report?'),
        ('Book a plane ticket for me.', None, 'I cannot book tickets with these tools, but I can help research flights.'),
        ('Erase the old one from my notes.', None, 'Which note should I delete?'),
        ('Read that website for me.', None, 'What URL should I read?'),
        ('Explain the difference between a list and a dictionary in Python.', None, 'A list is an ordered sequence; a dictionary maps keys to values.'),
    ]
    template = template_renderer()
    tokenizer = Tokenizer.from_file(str(ROOT/'research/tokenizer.json'))
    samples = []
    for prompt, name, args in cases:
        expected = {'role':'assistant','content':args if name is None else ''}
        if name:
            expected['tool_calls'] = [{'type':'function','function':{'name':name,'arguments':args}}]
            validate_calls(tools, expected['tool_calls'])
        row = {'tools':tools,'messages':[{'role':'user','content':prompt},expected],
               'source':'website-heldout-v1','source_id':digest(prompt)}
        samples.extend(render_samples(row, template, tokenizer))
    # Test that a tool result is used to answer, without endlessly calling again.
    row = {'tools':tools,'source':'website-heldout-v1','source_id':'tool-result-grounding',
           'messages':[{'role':'user','content':'Calculate 73 times 19.'},
           {'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'calculator','arguments':{'expression':'73 * 19'}}}]},
           {'role':'tool','content':'{"expression":"73 * 19","result":"1387","value":1387}'},
           {'role':'assistant','content':'73 times 19 is 1387.'}]}
    samples.extend(render_samples(row, template, tokenizer)[-1:])
    path = ROOT/'data/website-test.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in samples),encoding='utf-8')
    print(f'Wrote {len(samples)} held-out website probes to {path}')

if __name__ == '__main__':
    main()
