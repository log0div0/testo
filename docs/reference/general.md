# Test scripts general overview

At the global level, Testo language could be considered as a list of declarations. At the same top level there is also the `include` directive.

Possible declarations:

-   `machine` - virtual machine declaration.
-   `flash` - virtual flash drive declaration.
-   `network` - virtual network declaration.
-   `param` - param declaration.
-   `test` - test declaration.
-   `macro` - macro declaration.
-	Calling macros with statements (see [here](macro#macros-with-declarations))

Declarations may be placed in any convenient order and can be distributed between several files. Nonetheless, virtual machines, virtual flash drives, virtual networks, params and macros must be declared before being referenced. You can insert all the declarations from another file with the `include` directive. The `include` directive must be placed at the top level between declarations, not inside declarations.

The main concept of test composition looks like this. The tests developer declares all the virtual resources he or she would like to see in the current test bench. The declaration is done with the the `machine`, `flash` and `network` directives, followed by the attributes (specifications) for those resources. With the `param` the developer declares all the global string constants required for convenient tests developing. After that the developer moves on to the tests themselves.

Tests' bodies are basically a list of commands. Each command has a header, which is a virtual machine (or a virtual flash drive) name, and a body, which is an action (or an action block) to be applied to this virtual machine or flash drive. If the action (or one of the actions from the action block) fails, the whole test is considered failed and control moves on to the next queued test (if there is no `--stop_on_fail` command line argument specified).

> Conditions and cycles are also considered actions and could be used inside a command body.

Some of the most often used actions or commands can be grouped into a macro with the `macro` declaration. Macros can have parameters.
