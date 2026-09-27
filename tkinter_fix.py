import os
import sys

# Fix Tcl/Tk paths for virtual environments on Windows (Pip cannot download Tk UI library)
if sys.platform == 'win32' and sys.prefix != sys.base_prefix:
    tcl_dir = os.path.join(sys.base_prefix, 'tcl')
    tcl_lib = os.path.join(tcl_dir, 'tcl8.6')
    tk_lib = os.path.join(tcl_dir, 'tk8.6')
    if os.path.exists(tcl_lib) and 'TCL_LIBRARY' not in os.environ:
        os.environ['TCL_LIBRARY'] = tcl_lib
    if os.path.exists(tk_lib) and 'TK_LIBRARY' not in os.environ:
        os.environ['TK_LIBRARY'] = tk_lib

os.environ.setdefault('TK_SILENCE_DEPRECATION', '1')

import tkinter

# Only needs to be imported once at the beginning of the application
def apply_patch():
    # Create a monkey patch for the internal _tkinter module
    original_init = tkinter.Tk.__init__
    
    def patched_init(self, *args, **kwargs):
        # Call the original init
        original_init(self, *args, **kwargs)
        
        # Define the missing ::tk::ScreenChanged procedure
        self.tk.eval("""
        if {[info commands ::tk::ScreenChanged] == ""} {
            proc ::tk::ScreenChanged {args} {
                # Do nothing
                return
            }
        }
        """)
    
    # Apply the monkey patch
    tkinter.Tk.__init__ = patched_init

# Apply the patch automatically when this module is imported
apply_patch() 