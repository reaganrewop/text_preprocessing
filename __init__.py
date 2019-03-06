import sys
if not not globals(  ).get('init_modules'):
    # second or subsequent run: remove all but initially loaded modules
    for m in sys.modules.keys(  ):
        if m not in init_modules:
            del(sys.modules[m])
else:
    # first run: find out which modules were initially loaded
    init_modules = sys.modules.keys(  )
