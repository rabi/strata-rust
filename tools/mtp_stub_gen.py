# Generate abort stubs for the strata:: symbols the corpus leaves unresolved, so a call is loud.
# The CUDA runtime seam is faked with --wrap instead and its stubs just return cudaSuccess.
import subprocess, os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
objs = [os.path.join(R, 'target/corpus', o) for o in ('mc.o', 'mtp.o', 'layer.o')]
def undef(o):
    return set(l.split()[-1] for l in subprocess.run(['nm','-u',o],capture_output=True,text=True).stdout.splitlines() if l.strip())
def defined(o):
    s=set()
    for l in subprocess.run(['nm',o],capture_output=True,text=True).stdout.splitlines():
        p=l.split()
        if len(p)==3 and p[1] in 'TtWwVvDBbRr': s.add(p[2])
    return s
u=set().union(*[undef(o) for o in objs]); d=set().union(*[defined(o) for o in objs])
need=sorted(x for x in u-d if x.startswith('cuda') or 'strata' in x or 'kernels' in x)
out=['.text']
for i,s in enumerate(need):
    out += [f'.globl {s}', f'.type {s}, @function', f'{s}:']
    if s.startswith('cuda'):
        out += ['    xor %eax, %eax', '    ret']
    else:
        out += [f'    lea .Lm{i}(%rip), %rdi', '    call strata_stub_report@PLT', '    ud2']
out.append('.section .rodata')
for i,s in enumerate(need):
    if not s.startswith('cuda'):
        out.append(f'.Lm{i}: .asciz "{s}"')
open(os.path.join(R,'tools/mtp_stub.s'),'w').write('\n'.join(out)+'\n')
print("stubs:", len(need))
