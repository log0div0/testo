
import React from 'react'
import Terminal from '../../../server/components/Terminal'

export const terminal1 = (
	<Terminal height="600px">
		<span className="">C:\Users\Testo&gt; testo run ./tests.testo --stop_on_fail --param ISO_DIR C:\iso --test_spec cycles_demo<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="magenta ">server_install_guest_additions<br/></span>
		<span className="magenta ">server_install_vifm<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">cycles_demo<br/></span>
		<span className="blue ">[ 80%] Preparing the environment for test </span>
		<span className="yellow ">cycles_demo<br/></span>
		<span className="blue ">[ 80%] Restoring snapshot </span>
		<span className="yellow ">server_install_vifm</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Running test </span>
		<span className="yellow ">cycles_demo<br/></span>
		<span className="blue ">[ 80%] Typing </span>
		<span className="yellow ">"vifm /" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 2s<br/></span>
		<span className="blue ">[ 80%] Calling macro </span>
		<span className="yellow ">vifm_select(</span>
		<span className="yellow ">menu_entry="usr"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">G </span>
		<span className="blue ">2 times </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('usr').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] </span>
		<span className="yellow ">server</span>
		<span className="blue ">: Got to entry usr in 19 steps<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Calling macro </span>
		<span className="yellow ">vifm_select(</span>
		<span className="yellow ">menu_entry="sbin"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('sbin').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">G </span>
		<span className="blue ">2 times </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('sbin').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('sbin').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('sbin').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('sbin').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('sbin').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('sbin').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] Checking </span>
		<span className="yellow ">"find_text().match('sbin').match_background('blue').size() == 1"</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 80%] </span>
		<span className="yellow ">server</span>
		<span className="blue ">: Got to entry sbin in 6 steps<br/></span>
		<span className="blue ">[ 80%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="red bold">C:/Users/Testo/tests.testo:50:3: Caught abort action on virtual machine server with message: stop here<br/><br/>C:/Users/Testo/declarations.testo:15:1: note: the virtual machine server was declared here<br/><br/></span>
		<span className="red bold">[100%] Test </span>
		<span className="yellow bold">cycles_demo</span>
		<span className="red bold"> FAILED in 0h:0m:37s<br/></span>
		<span className="">C:\Users\Testo&gt; </span>
	</Terminal>
)

export const terminal2 = (
	<Terminal height="600px">
		<span className="">user$ sudo testo run ./ --stop_on_fail --param ISO_DIR /opt/iso --test_spec cycles_demo<br/></span>
		<span className="blue bold">UP-TO-DATE TESTS:<br/></span>
		<span className="magenta ">server_install_ubuntu<br/></span>
		<span className="magenta ">server_prepare<br/></span>
		<span className="blue bold">TESTS TO RUN:<br/></span>
		<span className="magenta ">cycles_demo<br/></span>
		<span className="blue ">[ 67%] Preparing the environment for test </span>
		<span className="yellow ">cycles_demo<br/></span>
		<span className="blue ">[ 67%] Restoring snapshot </span>
		<span className="yellow ">server_prepare</span>
		<span className="blue "> for virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Running test </span>
		<span className="yellow ">cycles_demo<br/></span>
		<span className="blue ">[ 67%] Typing </span>
		<span className="yellow ">"vifm /" </span>
		<span className="blue ">with interval 30ms in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 2s<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">19 times </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Calling macro </span>
		<span className="yellow ">vifm_select(</span>
		<span className="yellow ">menu_entry="usr"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Checking </span>
		<span className="yellow ">find_text('usr').match_background('blue').size() == 1 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] </span>
		<span className="yellow ">server</span>
		<span className="blue ">: Entry usr is already selected<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Calling macro </span>
		<span className="yellow ">vifm_select(</span>
		<span className="yellow ">menu_entry="sbin"</span>
		<span className="yellow ">)</span>
		<span className="blue "> in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Checking </span>
		<span className="yellow ">find_text('sbin').match_background('blue').size() == 1 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">G </span>
		<span className="blue ">2 times </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Checking </span>
		<span className="yellow ">find_text('sbin').match_background('blue').size() == 1 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 50ms<br/></span>
		<span className="blue ">[ 67%] Checking </span>
		<span className="yellow ">find_text('sbin').match_background('blue').size() == 1 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 50ms<br/></span>
		<span className="blue ">[ 67%] Checking </span>
		<span className="yellow ">find_text('sbin').match_background('blue').size() == 1 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 50ms<br/></span>
		<span className="blue ">[ 67%] Checking </span>
		<span className="yellow ">find_text('sbin').match_background('blue').size() == 1 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 50ms<br/></span>
		<span className="blue ">[ 67%] Checking </span>
		<span className="yellow ">find_text('sbin').match_background('blue').size() == 1 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 50ms<br/></span>
		<span className="blue ">[ 67%] Checking </span>
		<span className="yellow ">find_text('sbin').match_background('blue').size() == 1 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">DOWN </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] Sleeping in virtual machine </span>
		<span className="yellow ">server</span>
		<span className="blue "> for 50ms<br/></span>
		<span className="blue ">[ 67%] Checking </span>
		<span className="yellow ">find_text('sbin').match_background('blue').size() == 1 </span>
		<span className="blue ">in virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="blue ">[ 67%] </span>
		<span className="yellow ">server</span>
		<span className="blue ">: Got to entry sbin in 6 steps<br/></span>
		<span className="blue ">[ 67%] Pressing key </span>
		<span className="yellow ">ENTER </span>
		<span className="blue ">on virtual machine </span>
		<span className="yellow ">server<br/></span>
		<span className="red bold">/home/alex/testo/tests.testo:42:3: Caught abort action on virtual machine server with message: stop here<br/></span>
		<span className="red bold">[100%] Test </span>
		<span className="yellow bold">cycles_demo</span>
		<span className="red bold"> FAILED in 0h:0m:13s<br/></span>
		<span className="">user$ </span>
	</Terminal>
)
