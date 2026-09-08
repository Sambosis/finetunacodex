"""Build an attributed terminal SFT pilot from existing public examples.

No downloaded command is executed. PowerShell AST and bash -n checks are syntax-only.
Output is isolated from data/train.jsonl and the existing paid training launcher.
"""
import collections
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer
from prepare_data import ROOT, MODEL, MODEL_REVISION, canonical, digest, render_samples, template_renderer, validate_calls

OUT = ROOT/'data'/'terminal-v1'
RAW = OUT/'sources'
SEED = 3407
TOOLS = [{'type':'function','function':{'name':'run_terminal','description':'Execute a command in the specified shell on the corresponding operating system. Returns stdout, stderr and exit status. Use the current working directory when working_directory is omitted.', 'parameters':{'type':'object','properties':{'shell':{'type':'string','enum':['powershell','windows_powershell','bash']},'command':{'type':'string'},'working_directory':{'type':'string'}},'required':['shell','command'],'additionalProperties':False}}}]
BLOCK = re.compile(r'^```([^\n]*)\n(.*?)^```\s*$',re.M|re.S)
PARAM = re.compile(r'^### -([\w]+)', re.M)
COMMON_PARAMS = set('Verbose Debug ErrorAction WarningAction InformationAction ProgressAction ErrorVariable WarningVariable InformationVariable OutVariable OutBuffer PipelineVariable WhatIf Confirm'.lower().split())
ALIASES = dict(zip('gci dir ls gi cat gc sc sp gp ni ri rm del erase cp copy mv move ren sl cd pwd echo sort select where foreach ft fl measure sleep gps ps kill spps gsv sasv spsv ipmo epcsv ipcsv iex'.split(), 'Get-ChildItem Get-ChildItem Get-ChildItem Get-Item Get-Content Get-Content Set-Content Set-ItemProperty Get-ItemProperty New-Item Remove-Item Remove-Item Remove-Item Remove-Item Copy-Item Copy-Item Move-Item Move-Item Rename-Item Set-Location Set-Location Get-Location Write-Output Sort-Object Select-Object Where-Object ForEach-Object Format-Table Format-List Measure-Object Start-Sleep Get-Process Get-Process Stop-Process Stop-Process Get-Service Start-Service Stop-Service Import-Module Export-Csv Import-Csv Invoke-Expression'.split()))
AUTO_VARS = set('_ psitem true false null env:temp env:tmp env:path env:userprofile env:windir env:systemroot env:computername env:programfiles home pwd psversiontable executioncontext error args input matches lastexitcode pscommandpath psscriptroot host psulture psculture psuiculture pid psedition psdebugcontext shellid nestedpromptlevel'.split())
BASH_COMMANDS = set('find ls cp mv mkdir rmdir touch cat head tail sort uniq wc cut tr sed awk grep rg chmod chown chgrp stat du df ln readlink realpath basename dirname file tar gzip gunzip zip unzip sha256sum md5sum diff cmp tee printf echo pwd date env printenv ps pgrep pkill kill nice renice top free uptime uname id whoami groups getent hostname hostnamectl timedatectl journalctl systemctl ss ip ping curl wget dig host nslookup timeout xargs test true false sleep git python python3 pip pip3 node npm npx dotnet jq apt apt-get apt-cache dpkg dpkg-query dnf rpm which command type export set read'.split())
BAD = re.compile(r'(?i)(mimikatz|credential.dump|reverse.shell|Invoke-Expression|\biex\b|ExecutionPolicy\s+Bypass|EncodedCommand|DisableRealtimeMonitoring|Clear-Disk|Format-Volume|Remove-Partition|bcdedit|cipher\s+/w|rm\s+-[rf]+\s+/(?:\s|$)|curl[^\n]*\|\s*(?:bash|sh)|wget[^\n]*\|\s*(?:bash|sh))')

def jsonl(path, rows):
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')

def readjsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s]

def clean_prose(text):
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)',r'\1',text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*',r'\1',text)
    return re.sub(r'\s+', ' ',text).strip()

def topic(command, module=''):
    c=command.lower()
    if re.search(r'hk(?:lm|cu|cr|u):|registry::|registrykey',c): return 'registry'
    if 'scheduled' in c or module.lower() in ['scheduledtasks','psscheduledjob']: return 'scheduled_tasks'
    if module.lower() in ['netadapter','nettcpip','netsecurity'] or re.search(r'\b(?:ss|ip|ping|dig|curl|wget)\b',c): return 'network'
    if module.lower() in ['storage','dism','pnpdevice'] or 'acl' in c or re.search(r'\b(?:chmod|chown|chgrp)\b',c): return 'os_configuration'
    if module.lower() in ['appx','packagemanagement','powershellget'] or re.search(r'\b(?:git|npm|node|python3?|pip3?|dotnet|apt|dpkg|dnf)\b',c): return 'applications'
    if re.search(r'process|service|eventlog|winevent|\b(?:systemctl|journalctl|ps|kill|pgrep|pkill)\b',c): return 'processes_services_logs'
    if re.search(r'item|content|path|file|csv|json|archive|\b(?:find|ls|cp|mv|cat|grep|sed|awk|tar|sort|uniq)\b',c): return 'files_data'
    return 'shell_scripting'

def document_candidates(lock,rejected):
    docs=[]
    with tarfile.open(RAW/'powershell-docs.tar.gz') as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith('.md'): continue
            path=member.name.split('/',1)[1]
            if path.startswith('reference/7.4/') or path.startswith('reference/5.1/'):
                if '/About/' not in path:
                    docs.append(('MicrosoftDocs/PowerShell-Docs',lock['docs_revision'],path,archive.extractfile(member).read().decode('utf-8-sig')))
            if path in ['LICENSE.md','LICENSE-CODE.md']:
                (RAW/('powershell-'+path)).write_bytes(archive.extractfile(member).read())
    for file in (RAW/'windows-docs'/'docset').rglob('*.md'):
        docs.append(('MicrosoftDocs/windows-powershell-docs',lock['windows_docs_revision'],file.relative_to(RAW/'windows-docs').as_posix(),file.read_text(encoding='utf-8-sig')))
    inventory={}
    for repo,rev,path,text in docs:
        name=Path(path).stem.lower()
        inventory.setdefault(name,set()).update(p.lower() for p in PARAM.findall(text))
    rows=[]
    # Prefer current PowerShell examples over duplicate 5.1 examples.
    docs.sort(key=lambda d:('reference/7.4/' not in d[2],d[2]))
    for repo,rev,path,text in docs:
        module=path.split('/')[-2]
        if not re.search(r'^## EXAMPLES\s*$',text,re.M|re.I): continue
        section=re.split(r'^## EXAMPLES\s*$',text,flags=re.M|re.I)[1]
        section=re.split(r'^## ',section,flags=re.M)[0]
        examples=re.split(r'^### (?:Example\s*\d*\s*[:\-]?\s*|-*\s*EXAMPLE\s*\d*\s*-*\s*)',section,flags=re.M|re.I)[1:]
        for index,example in enumerate(examples,1):
            blocks=list(BLOCK.finditer(example))
            codeblocks=[b for b in blocks if b[1].strip().lower() in ['powershell','ps','ps1'] or (not b[1].strip() and re.search(r'^PS [^>\n]*>',b[2],re.M))]
            if len(codeblocks)!=1:
                rejected['docs_not_single_code_block']+=1; continue
            block=codeblocks[0]
            command=block[2].strip()
            if not block[1].strip():
                command='\n'.join(re.findall(r'^PS [^>\n]*>\s?(.*)$',command,re.M))
            command=re.sub(r'^PS [^>\n]*>\s?', '',command, flags=re.M)
            title=example.split('\n',1)[0].strip(' -')
            prose=clean_prose(example[len(example.split('\n',1)[0]):block.start()])
            if not prose:
                prose=clean_prose(BLOCK.sub('',example[block.end():]))
            # Exclude dependency on omitted examples, interactive transcripts and ellipses.
            if not prose or re.search(r'previous example|preceding example|shown above|earlier example|following examples|\[!|\.{3}',prose,re.I) or len(prose)>1800:
                rejected['docs_missing_or_dependent_context']+=1; continue
            if re.search(r'\.{3}|<[^\n>]+>|^>>|^PS>',command,re.M) or not command or len(command)>3500:
                rejected['docs_placeholder_or_transcript']+=1; continue
            if BAD.search(command) or re.search(r'/(?:etc|usr|var|home)/',command):
                rejected['docs_outside_windows_pilot']+=1; continue
            shell='powershell' if '/7.4/' in path else 'windows_powershell'
            row={'id':digest([repo,path,index]),'instruction':title+'. '+prose,'command':command,
                 'shell':shell,'os':'windows','source':repo,'revision':rev,'source_id':path+f'#example-{index}',
                 'source_url':f'https://github.com/{repo}/blob/{rev}/{path}',
                 'license_spdx':'CC-BY-4.0 (prose); MIT (code)','module':module,
                 'category':topic(command,module),'origin_group':path.split('/')[-1].lower(),
                 'transformation':'One existing example code block paired with its preceding description; markdown and console prompts stripped. No generated output.',
                 'validation_level':'source_documented_not_executed'}
            rows.append(row)
    return rows,inventory

def cli_candidates(lock,rejected):
    rows=[]; counts=collections.Counter(); native_ps=0
    audit=[]
    for file in sorted((RAW/'cli'/'data'/'default'/'train').glob('*.parquet')):
        for batch in pq.ParquetFile(file).iter_batches(batch_size=8192):
            for r in batch.to_pylist():
                if r['language']!='en' or r['shell'] not in ['powershell','bash']: continue
                counts[r['shell']]+=1
                user=next((m['content'] for m in r['messages'] if m['role']=='user'),'')
                command=r['messages'][-1]['content'].strip()
                if r['shell']=='powershell':
                    native_ps+=bool(re.search(r'\b(?:Get|Set|New|Remove|Start|Stop|Copy|Move|Test|Add|Clear|Rename|Import|Export|Register|Unregister|Enable|Disable)-[A-Z][a-z]',command))
                    rejected['cli_powershell_rewrite_excluded']+=1
                    if len(audit)<200: audit.append({'id':r['id'],'instruction':user,'command':command,'source':r['source']})
                    continue
                # Imported curated examples only: exclude bulk synthesis, conversions and tldr placeholder corruption.
                if not r['source'].startswith('caro-eval-curated') or r['quality_tier']!='imported':
                    rejected['cli_noncurated_source']+=1; continue
                first=command.split()[0] if command else ''
                if first not in BASH_COMMANDS or BAD.search(command) or re.search(r'\b(?:brew|osascript|pbcopy|launchctl|defaults)\b|/Users/|\{\{|<[^>]+>|YOUR_|\.{3}',command):
                    rejected['cli_scope_or_placeholder']+=1; continue
                if len(command)>1600 or not user or 'security' in (r['bucket'] or []):
                    rejected['cli_outside_pilot']+=1; continue
                rows.append({'id':digest(['cli',r['id']]),'instruction':user,'command':command,'shell':'bash','os':'linux',
                    'source':'carosh/cli-1m','revision':lock['cli_revision'],'source_id':r['id'],
                    'source_url':f"https://huggingface.co/datasets/carosh/cli-1m/tree/{lock['cli_revision']}",
                    'license_spdx':r['license_spdx'],'upstream_source':r['source'],
                    'category':topic(command),'module':first,
                    'origin_group':re.sub(r'-\d+$','',r['id']),
                    'transformation':'Original English curated instruction and Bash command, wrapped as a terminal tool call.',
                    'validation_level':'source_curated_not_executed'})
    jsonl(OUT/'audit-cli-powershell.jsonl',audit)
    return rows,{'english_counts':dict(counts),'native_powershell_regex_matches':native_ps}

def parse_powershell(rows,inventory,rejected):
    result=[]
    for shell,exe in [('powershell',shutil.which('pwsh')),('windows_powershell',shutil.which('powershell'))]:
        group=[r for r in rows if r['shell']==shell]
        inputpath=OUT/(shell+'-parse-input.jsonl'); outputpath=OUT/(shell+'-parse-results.jsonl')
        jsonl(inputpath,group)
        subprocess.run([exe,'-NoProfile','-NonInteractive','-File',str(ROOT/'inspect_terminal_ps.ps1'),'-InputPath',str(inputpath),'-OutputPath',str(outputpath)],check=True,timeout=180)
        reports={r['id']:r for r in readjsonl(outputpath)}
        for row in group:
            report=reports[row['id']]
            reason=None
            if report['errors']: reason='powershell_parse_error'
            used={v.lower() for v in report['variables']}
            defined={v.lower() for v in report['defined']}
            missing=used-defined-AUTO_VARS
            if missing: reason='powershell_undefined_variable'
            for cmd in report['commands']:
                name=cmd['name']
                if not name: reason='powershell_dynamic_command'; continue
                name=ALIASES.get(name.lower(),name).lower()
                if name not in inventory:
                    reason='powershell_external_or_unknown_command'; continue
                for param in cmd['parameters']:
                    p=param.lower()
                    possible=inventory[name]|COMMON_PARAMS
                    if p not in possible and len([x for x in possible if x.startswith(p)])!=1:
                        reason='powershell_unknown_parameter'
            if reason:
                rejected[reason]+=1
                row['rejection_reason']=reason
            else:
                row['syntax_parser']=report['parser_version']; result.append(row)
    return result

def parse_bash(rows,rejected):
    # bash -n parses stdin without executing expansions, commands or redirections.
    result=[]
    for row in rows:
        p=subprocess.run(['wsl.exe','--exec','bash','--noprofile','--norc','-n'],input=row['command']+'\n',encoding='utf-8',capture_output=True,timeout=20)
        if p.returncode: rejected['bash_parse_error']+=1
        else: row['syntax_parser']='GNU bash -n (WSL)'; result.append(row)
    return result

def tldr_candidates(lock,rejected):
    rows=[]
    with tarfile.open(RAW/'tldr.tar.gz') as archive:
        for m in archive.getmembers():
            if not m.isfile(): continue
            path=m.name.split('/',1)[1]
            if path in ['LICENSE.md','LICENSE']:
                (RAW/'tldr-LICENSE.md').write_bytes(archive.extractfile(m).read())
            if not path.endswith('.md') or not (path.startswith('pages/common/') or path.startswith('pages/linux/')): continue
            # Direct source examples for a bounded set of file, process, OS and app tools.
            stem=Path(path).stem
            if stem.split('-')[0] not in BASH_COMMANDS: continue
            text=archive.extractfile(m).read().decode('utf-8-sig')
            pairs=re.findall(r'^- (.*?)\n\s*\n`([^\n]+)`',text,re.M)
            for index,(description,template) in enumerate(pairs,1):
                bindings={}
                def replace(match):
                    placeholder=match[1]
                    # tldr's bracketed short/long option alternatives are notation, not literal flags.
                    if placeholder.startswith('[') and placeholder.endswith(']'):
                        value=placeholder[1:-1].split('|')[0]
                    else: value=placeholder.split('|')[0]
                    value=re.sub(r'\s+\.\.\.$','',value)
                    bindings[placeholder]=value
                    return value
                command=re.sub(r'\{\{(.*?)\}\}',replace,template)
                if BAD.search(command) or re.search(r'\.{3}|<[^>]+>|YOUR_|\{\{|\b(?:osascript|pbcopy|launchctl)\b',command) or len(command)>1200:
                    rejected['tldr_unsupported_template']+=1; continue
                if stem=='systemctl' and index==1:
                    rejected['tldr_known_semantic_mismatch']+=1; continue
                # Keep literal fixture names and document all chosen alternatives in the input.
                instruction=description.rstrip(':')+'.'
                if bindings:
                    instruction+=' Use these example arguments and choices: '+ '; '.join(f'{k} = {v}' for k,v in bindings.items())+'.'
                rows.append({'id':digest(['tldr',path,index]),'instruction':instruction,'command':command,
                    'shell':'bash','os':'linux','source':'tldr-pages/tldr','revision':lock['tldr_revision'],
                    'source_id':path+f'#example-{index}','source_url':f"https://github.com/tldr-pages/tldr/blob/{lock['tldr_revision']}/{path}",
                    'license_spdx':'CC-BY-4.0','module':stem,'category':topic(command),
                    'origin_group':stem.split('-')[0],
                    'transformation':'Existing English description and command; tldr braces removed, first alternative selected, every choice stated in user input. No generated output.',
                    'validation_level':'source_documented_not_executed'})
    return rows

def normalized(text):
    text=text.lower()
    text=re.sub(r'\b(?:in|using) powershell\b','',text)
    text=re.sub(r'\d+','#',text)
    return re.sub(r'\W+',' ',text).strip()

def assign_groups(rows):
    # Union exact commands, source families, and normalized instructions across sources.
    parents=list(range(len(rows)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    seen={}
    for i,r in enumerate(rows):
        keys=[('source_family',r['source'],r['origin_group']),('command',normalized(r['command'])),('instruction',normalized(r['instruction']))]
        for key in keys:
            if key in seen: parents[find(i)]=find(seen[key])
            else: seen[key]=i
    members=collections.defaultdict(list)
    for i,r in enumerate(rows): members[find(i)].append(r['id'])
    for i,r in enumerate(rows):
        r['group_id']=digest(sorted(members[find(i)]))
        slot=int(r['group_id'][:8],16)%100
        r['split']='test' if slot<10 else 'validation' if slot<20 else 'train'

def render(row,template,tokenizer):
    call={'role':'assistant','content':'','tool_calls':[{'type':'function','function':{'name':'run_terminal','arguments':{'shell':row['shell'],'command':row['command']}}}]}
    validate_calls(TOOLS,call['tool_calls'])
    version='PowerShell 7.4 on Windows' if row['shell']=='powershell' else 'Windows PowerShell 5.1 on Windows' if row['shell']=='windows_powershell' else 'Bash on Linux'
    # Existing examples can refer to fixture files and external apps. State that context explicitly.
    system=f'You operate a terminal using structured tool calls. Environment: {version}. The task is a documentation-based exercise in a prepared disposable environment. Referenced example files, accounts, services, and required modules or applications are provisioned; use the current directory for relative paths. The user authorizes the described task only. Return the command through run_terminal. Do not claim a successful result before observing tool output.'
    messages=[{'role':'system','content':system},{'role':'user','content':row['instruction']},call]
    samples=render_samples({'source':row['source'],'source_id':row['source_id'],'messages':messages,'tools':TOOLS},template,tokenizer)
    if not samples: return None
    sample=samples[0]
    for key,value in row.items():
        if key not in ['instruction','command']: sample[key]=value
    sample['supervised_tokens']=len(tokenizer.encode(sample['completion'],add_special_tokens=False).ids)
    return sample

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=json.loads((OUT/'source-lock.json').read_text())
    rejected=collections.Counter()
    docs,inventory=document_candidates(lock,rejected)
    cli,audit=cli_candidates(lock,rejected)
    cli+=tldr_candidates(lock,rejected)
    print('Candidates',len(docs),'Windows',len(cli),'Linux',flush=True)
    jsonl(OUT/'candidates.jsonl',docs+cli)
    accepted=parse_powershell(docs,inventory,rejected)+parse_bash(cli,rejected)
    dedup=[]; seen=set()
    for row in accepted:
        key=(row['os'],re.sub(r'\s+',' ',row['command']).lower())
        if key in seen: rejected['exact_command_duplicate']+=1; continue
        seen.add(key); dedup.append(row)
    assign_groups(dedup)
    template=template_renderer(); tokenizer=Tokenizer.from_file(str(ROOT/'research/tokenizer.json'))
    buckets=collections.defaultdict(list)
    for row in dedup:
        sample=render(row,template,tokenizer)
        if sample: buckets[row['split']].append(sample)
        else: rejected['overlength']+=1
    manifest={'name':'terminal-v1','stage':'single_turn_command_sft_pilot','base_model':MODEL,'model_revision':MODEL_REVISION,'max_length':4096,'seed':SEED,
              'source_revisions':{k:v for k,v in lock.items() if k.endswith('revision')},'cli_audit':audit,'rejected':dict(rejected),'splits':{},
              'execution_validation':'None of the downloaded commands were executed. AST/parameter and bash -n checks are static, not proof of task completion.',
              'split_method':'Connected components of documentation page/source family, normalized command and normalized instruction; 80/10/10 stable group hash before selection.',
              'terminal_tool_contract':'run_terminal is a new proposed deployment contract, not a claim that the existing website implements it.',
              'limitations':['Documentation exercises assume provisioned targets and dependencies.','No multi-step tool observations or recovery trajectories.','Semantic correctness is source-derived and spot-reviewed, not exhaustively execution-tested.','Exact/normalized grouping reduces but cannot eliminate semantic overlap or pretraining contamination.']}
    jsonl(OUT/'accepted-pool.jsonl',[r for rows in buckets.values() for r in rows])
    rng=random.Random(SEED)
    for split in ['train','validation','test']:
        windows=[r for r in buckets[split] if r['os']=='windows']; linux=[r for r in buckets[split] if r['os']=='linux']
        rng.shuffle(windows); rng.shuffle(linux)
        # Retain Windows variety; cap Linux to at most 40% of terminal rows.
        linux=linux[:int(len(windows)*2/3)]
        terminal=windows+linux
        replay=[]
        if split=='train':
            original=readjsonl(ROOT/'data'/'train.jsonl'); rng.shuffle(original)
            desired=int(len(terminal)*.15/.85)
            win_tokens=sum(r['supervised_tokens'] for r in windows)
            other_tokens=sum(r['supervised_tokens'] for r in linux)
            for r in original:
                if len(replay)>=desired: break
                r=dict(r); n=len(tokenizer.encode(r['completion'],add_special_tokens=False).ids)
                if other_tokens+n>win_tokens: continue
                r.update({'os':'retention','shell':'original_tools','category':'tool_retention','supervised_tokens':n,'split':'train','license_spdx':'Apache-2.0','group_id':'retention-'+r['source_id']})
                replay.append(r); other_tokens+=n
            # Windows must be the majority even when counting replay rows.
            replay=replay[:max(0,2*len(windows)-len(terminal))]
        rows=terminal+replay; rng.shuffle(rows)
        jsonl(OUT/(split+'.jsonl'),rows)
        path=OUT/(split+'.jsonl')
        manifest['splits'][split]={'rows':len(rows),'by_os':dict(collections.Counter(r['os'] for r in rows)),
            'by_shell':dict(collections.Counter(r['shell'] for r in rows)),'by_category':dict(collections.Counter(r['category'] for r in rows)),
            'by_source':dict(collections.Counter(r['source'] for r in rows)),
            'tokens':sum(r['tokens'] for r in rows),'supervised_tokens':sum(r['supervised_tokens'] for r in rows),
            'windows_row_share':len(windows)/len(rows),'windows_supervised_token_share':sum(r['supervised_tokens'] for r in windows)/sum(r['supervised_tokens'] for r in rows),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        assert manifest['splits'][split]['windows_row_share']>=.5
        assert manifest['splits'][split]['windows_supervised_token_share']>=.5
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (OUT/'terminal-tools.json').write_text(json.dumps(TOOLS,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
