import inspect, time, warnings

IGNORE = ['def ', 'for ', '#', '@', 'print ', 'print(']

def profile(external_globals=None):

    def as_decorator(fn):
        if external_globals:
            globals().update(external_globals)
        source, firstlineno = inspect.getsourcelines(fn)
        source = [line.strip('\n') for line in source]
        padded_length = max(len(line) for line in source) + 5

        profiled_source_lines = []
        psl_append = lambda string: profiled_source_lines.append(indent+string)
        for indented_line in source[1:]:
            line = indented_line.strip()

            if not line or any(line.startswith(i) for i in IGNORE):
                profiled_source_lines.append(indented_line)
                continue

            indent = ' '*(len(indented_line)-len(line))
            psl_append("start = time.time()")

            if line.startswith("return "):
                psl_append(line.replace("return ", "out = "))
            else:
                psl_append(line)
            psl_append("time_taken = 1000*(time.time()-start)")
            psl_append("print( '{}{{time_taken:>10.0f}}ms'.format(**locals()) )"\
                       .format(indented_line.ljust(padded_length)))
            if line.startswith("return "):
                psl_append("return out")
        
        profiled_source = '\n'.join(profiled_source_lines)
        try:
            exec profiled_source in globals()
            return eval(fn.func_name)
        except IndentationError as e:
            print( profiled_source )
            raise(e)
    
    return as_decorator