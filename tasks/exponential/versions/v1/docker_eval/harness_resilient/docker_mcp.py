"""Stdio MCP server exposing exactly one tool inside one pre-created Docker container.

No container creation, mounts, host file methods, Docker socket exposure or arbitrary
Docker arguments are offered to the model. stdout is protocol only.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import selectors
import signal
import subprocess
import sys
import time

TOOL = {'name':'exec','description':'Run a shell command in the isolated offline evaluation workspace. Use it to read public instructions, write code, inspect installed libraries and run local tests. Only /workspace and permitted temporary storage are writable. No network or host files are available. Output is bounded; longer computations should write artifacts and report a compact summary.','inputSchema':{'type':'object','properties':{'command':{'type':'string','minLength':1,'maxLength':65536},'timeout_seconds':{'type':'integer','minimum':1,'maximum':120,'default':60},'max_output_chars':{'type':'integer','minimum':100,'maximum':20000,'default':12000}},'required':['command'],'additionalProperties':False}}

class ArgumentValidationError(ValueError):
    """Only fixed schema diagnostics; never retain arbitrary arguments or values."""
    def __init__(self,code,field,expected,received_type):
        super().__init__('Workspace tool arguments failed validation.')
        self.detail={'category':'tool_argument_validation','code':code,'field':field,
                     'expected':expected,'received_type':received_type}

def value_type(value):
    if value is None:return 'null'
    if type(value) is bool:return 'boolean'
    if type(value) is int:return 'integer'
    if type(value) is float:return 'number'
    if isinstance(value,str):return 'string'
    if isinstance(value,list):return 'array'
    if isinstance(value,dict):return 'object'
    return 'other'

def validate_arguments(args):
    # Non-object arguments remain a protocol error, not a recoverable tool call.
    if not isinstance(args,dict):raise ValueError('Tool arguments must be an object.')
    if set(args)-{'command','timeout_seconds','max_output_chars'}:
        raise ArgumentValidationError('unexpected_fields','arguments',
            'Only command, timeout_seconds, max_output_chars are allowed.','object')
    command=args.get('command')
    if not isinstance(command,str) or not 1<=len(command)<=65536 or '\x00' in command:
        raise ArgumentValidationError('invalid_command','command',
            'A required string with 1..65536 characters and no NUL.',
            value_type(command) if 'command' in args else 'missing')
    timeout=args.get('timeout_seconds',60);limit=args.get('max_output_chars',12000)
    if type(timeout) is not int or not 1<=timeout<=120:
        raise ArgumentValidationError('invalid_timeout','timeout_seconds',
            'An integer in [1,120].',value_type(timeout))
    if type(limit) is not int or not 100<=limit<=20000:
        raise ArgumentValidationError('invalid_output_limit','max_output_chars',
            'An integer in [100,20000].',value_type(limit))
    return command,timeout,limit

def call_tool(workspace,name,arguments):
    """Validate before execute, including when workspace is a test double."""
    if name!='exec':raise ValueError('Unknown tool.')
    try:validate_arguments(arguments)
    except ArgumentValidationError as exc:
        return {'content':[{'type':'text','text':json.dumps(
            {'error':exc.detail,'command_executed':False},ensure_ascii=False)}],'isError':True}
    value=workspace.execute(arguments)
    return {'content':[{'type':'text','text':json.dumps(value,ensure_ascii=False)}],'isError':False}

def bounded_run(command,timeout,max_chars):
    """Drain bounded chunks, keep bounded memory, and enforce a host-side deadline."""
    process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','DOCKER_CONFIG')})
    selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
    deadline=time.monotonic()+timeout
    kept=bytearray();total=0;timed_out=False;limit_bytes=max_chars*4
    try:
        while selector.get_map():
            remaining=deadline-time.monotonic()
            if remaining<=0:
                timed_out=True
                try:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                break
            for key,_ in selector.select(min(remaining,.1)):
                chunk=os.read(key.fileobj.fileno(),65536)
                if not chunk:selector.unregister(key.fileobj);continue
                total+=len(chunk)
                kept.extend(chunk[:max(0,limit_bytes-len(kept))])
        try:code=process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            timed_out=True
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=process.wait(timeout=2)
    finally:
        selector.close();process.stdout.close()
    output=kept.decode('utf-8',errors='replace')
    truncated=total>len(kept) or len(output)>max_chars
    return {'exit_code':code,'timed_out':timed_out,'output':output[:max_chars],'output_truncated':truncated,'output_bytes_seen':total}

class DockerWorkspace:
    def __init__(self,docker,host,container):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}',container):raise ValueError('Invalid container identifier.')
        self.prefix=[docker,'--host',host];self.container=container
    def verify(self):
        r=subprocess.run(self.prefix+['inspect',self.container],capture_output=True,text=True,timeout=10,env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','DOCKER_CONFIG')})
        if r.returncode:raise RuntimeError('Configured workspace container is not available.')
        info=json.loads(r.stdout)[0];cfg=info['HostConfig']
        if cfg.get('NetworkMode')!='none' or cfg.get('Privileged') or cfg.get('PidMode')=='host' or cfg.get('IpcMode')=='host':raise RuntimeError('Container isolation configuration rejected.')
        if cfg.get('Memory',0)<=0 or cfg['Memory']>4*1024**3:raise RuntimeError('Container must have a memory limit <=4GiB.')
        if cfg.get('NanoCpus',0)<=0 or cfg['NanoCpus']>1000000000:raise RuntimeError('Container must have a CPU limit <=1 CPU.')
        if cfg.get('Binds') or any(x.get('Type')!='tmpfs' for x in info.get('Mounts',[])):raise RuntimeError('Host or named-volume files are not permitted in the workspace.')
        if not cfg.get('ReadonlyRootfs') or info.get('Config',{}).get('User')!='1000:1000':raise RuntimeError('Workspace requires a read-only root and user 1000:1000.')
        if set(cfg.get('CapDrop') or [])!={'ALL'} or cfg.get('CapAdd'):raise RuntimeError('Workspace must drop every Linux capability.')
        if not any(value.split(':')[0]=='no-new-privileges' for value in cfg.get('SecurityOpt',[])):raise RuntimeError('Workspace must forbid new privileges.')
        if cfg.get('Devices') or cfg.get('DeviceRequests'):raise RuntimeError('Workspace must not expose additional devices.')
        if cfg.get('MemorySwap')!=cfg['Memory']:raise RuntimeError('Workspace swap must not exceed its memory limit.')
        if not 0<cfg.get('PidsLimit',0)<=64:raise RuntimeError('Workspace process limit must be in [1,64].')
        if not info.get('State',{}).get('Running'):raise RuntimeError('Workspace container is not running.')
        if info['State'].get('Paused'):raise RuntimeError('Workspace container is already frozen.')
        return {'network_mode':cfg['NetworkMode'],'memory_bytes':cfg['Memory'],'nano_cpus':cfg['NanoCpus'],'privileged':False,'host_bind_mounts':False,'named_volumes':False,'readonly_root':True,'user':'1000:1000','all_capabilities_dropped':True,'no_new_privileges':True,'pids_limit':cfg['PidsLimit'],'memory_swap_bytes':cfg['MemorySwap']}
    def execute(self,args):
        command,timeout,limit=validate_arguments(args)
        argv=self.prefix+['exec','-i','--workdir','/workspace',self.container,'/usr/bin/timeout','--signal=TERM','--kill-after=2',str(timeout),'/bin/bash','-lc',command]
        return bounded_run(argv,timeout+5,limit)
    def freeze(self):
        """Maintainer-only stop hook; intentionally not exposed as an MCP tool."""
        started=time.monotonic()
        result=subprocess.run(self.prefix+['pause',self.container],capture_output=True,text=True,timeout=10,env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','DOCKER_CONFIG')})
        inspected=subprocess.run(self.prefix+['inspect','--format','{{.State.Paused}}',self.container],capture_output=True,text=True,timeout=10,env={k:v for k,v in os.environ.items() if k in ('PATH','HOME','DOCKER_CONFIG')})
        paused=inspected.returncode==0 and inspected.stdout.strip()=='true'
        return {'success':paused,'paused':paused,'pause_exit_code':result.returncode,'elapsed_seconds':time.monotonic()-started}

def serve(workspace):
    for line in sys.stdin.buffer:
        if len(line)>1024*1024:continue
        ident=None
        try:
            request=json.loads(line);ident=request.get('id');method=request.get('method');params=request.get('params') or {}
            if method=='initialize':result={'protocolVersion':params.get('protocolVersion','2024-11-05'),'capabilities':{'tools':{'listChanged':False}},'serverInfo':{'name':'frontieram-offline-workspace','version':'1.0'}}
            elif method=='tools/list':result={'tools':[TOOL]}
            elif method=='tools/call':
                result=call_tool(workspace,params.get('name'),params.get('arguments',{}))
            elif method=='ping':result={}
            elif method.startswith('notifications/'):continue
            else:
                if ident is not None:print(json.dumps({'jsonrpc':'2.0','id':ident,'error':{'code':-32601,'message':'Method not found'}}),flush=True)
                continue
            if ident is not None:print(json.dumps({'jsonrpc':'2.0','id':ident,'result':result},ensure_ascii=False),flush=True)
        except (ValueError,KeyError,TypeError,RuntimeError,OSError):
            if ident is not None:print(json.dumps({'jsonrpc':'2.0','id':ident,'error':{'code':-32602,'message':'Workspace request failed validation or execution.'}}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--docker',required=True);p.add_argument('--host',required=True);p.add_argument('--container',required=True);a=p.parse_args()
    w=DockerWorkspace(a.docker,a.host,a.container);w.verify();serve(w)
