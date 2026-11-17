"""Process identity helpers: stale PID records never authorize interruption."""
import psutil,time

def same_process(pid,birth):
 try:
  p=psutil.Process(int(pid))
  return p.status()!=psutil.STATUS_ZOMBIE and abs(p.create_time()-float(birth))<.01
 except (psutil.NoSuchProcess,TypeError,ValueError):return False

def stop_recorded(record):
 pid=record.get('pid');birth=record.get('birth')
 if not same_process(pid,birth):return False
 p=psutil.Process(pid);children=[]
 for q in p.children(recursive=True):
  try:children.append((q.pid,q.create_time()))
  except psutil.NoSuchProcess:pass
 for ident in children+[(pid,birth)]:
  if same_process(*ident):
   try:psutil.Process(ident[0]).terminate()
   except psutil.NoSuchProcess:pass
 return True
