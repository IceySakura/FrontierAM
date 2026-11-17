import sys
sys.path.insert(0, '/workspace/samples/support')
import api as P

def make_solver(d, seed, options):
    return Solver(d)

class Solver:
    def __init__(self, d):
        self.d = d
        self.n = 0
        self.log = open('/tmp/probe.log', 'w')
    def solve(self, hvp, b, context):
        self.n += 1
        self.log.write("SOLVE step=%d phase_changed=%s\n" % (context['step'], context['phase_changed']))
        self.log.flush()
        self.log.write("  context keys: %s\n" % sorted(context.keys()))
        self.log.flush()
        return b / float(self.d)
