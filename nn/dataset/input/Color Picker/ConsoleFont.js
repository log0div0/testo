
import React from 'react'
import path from 'path'
import {ConsoleTextLine, ConsoleText} from '../../common'
import {randomColors, randomInt} from '../../random'

function MyTextLine(props) {
	let colors = randomColors()
	return <ConsoleTextLine font='VGA16' color={colors.fg} backgroundColor={colors.bg}>{props.children}</ConsoleTextLine>
}

function MyText(props) {
	let colors = randomColors()
	return <ConsoleText font='VGA16' color={colors.fg} backgroundColor={colors.bg}>{props.children}</ConsoleText>
}

export function Example(props) {
	let bgPath = 'file://' + path.join(__dirname, 'desktop.png')

	return (
		<div id='background'>
			<MyTextLine>Placebo</MyTextLine>
			<MyTextLine>Nancy Boy</MyTextLine>
			<MyTextLine>Alcoholic kind of mood</MyTextLine>
			<MyTextLine>Lose my clothes</MyTextLine>
			<MyTextLine>Lose my <MyText>lube</MyText></MyTextLine>
			<MyTextLine><MyText>Cruising</MyText> for a piece of fun</MyTextLine>
			<MyTextLine>Looking out for number one</MyTextLine>
			<MyTextLine>Different partner every night</MyTextLine>
			<MyTextLine>So narcotic outta sight</MyTextLine>
			<MyTextLine>What a <MyText>gas</MyText></MyTextLine>
			<MyTextLine>what a beautiful ass</MyTextLine>
			<MyTextLine>And it all breaks down at the role <MyText>reversal</MyText></MyTextLine>
			<MyTextLine>Got the <MyText>muse</MyText> in my head she's universal</MyTextLine>
			<MyTextLine>Spinnin' me round she's coming over me</MyTextLine>
			<MyTextLine>And it all breaks down at the first <MyText>rehearsal</MyText></MyTextLine>
			<MyTextLine>Got the muse in my head she's universal</MyTextLine>
			<MyTextLine>Spinnin' me round she's coming over me</MyTextLine>
			<MyTextLine>Kind of buzz that lasts for days</MyTextLine>
			<MyTextLine>Had some help from <MyText>insect</MyText> ways</MyTextLine>
			<MyTextLine>Comes across all shy and <MyText>coy</MyText></MyTextLine>
			<MyTextLine>Just another nancy boy</MyTextLine>
			<MyTextLine>Woman man or modern monkey</MyTextLine>
			<MyTextLine>Just another happy <MyText>junkie</MyText></MyTextLine>
			<MyTextLine>Fifty pounds</MyTextLine>
			<MyTextLine>Press my button</MyTextLine>
			<MyTextLine>Going down</MyTextLine>
			<MyTextLine>And it all breaks down at the role reversal</MyTextLine>
			<MyTextLine>Got the muse in my head she's universal</MyTextLine>
			<MyTextLine><MyText>Spinnin'</MyText> me round she's coming over me</MyTextLine>
			<MyTextLine>And it all breaks down at the first rehearsal</MyTextLine>
			<MyTextLine>Got the muse in my head she's universal</MyTextLine>
			<MyTextLine>Spinnin' me round <MyText>she's</MyText> coming over me</MyTextLine>
			<MyTextLine>does his makeup in his room</MyTextLine>
			<MyTextLine><MyText>Douse</MyText> himself with cheap perfume</MyTextLine>
			<MyTextLine>Eyeholes in a paper bag</MyTextLine>
			<MyTextLine>Greatest lay I ever had</MyTextLine>
			<MyTextLine>Kind of guy who <MyText>mates</MyText> for life</MyTextLine>
			<MyTextLine>Gotta help him find a wife</MyTextLine>
			<MyTextLine>We're a couple</MyTextLine>
			<MyTextLine>When our bodies double</MyTextLine>
			<MyTextLine>And it <MyText>all breaks down</MyText> at the role reversal</MyTextLine>
			<MyTextLine>Got the muse in my head she's universal</MyTextLine>
			<MyTextLine>Spinnin' me round she's coming over me</MyTextLine>
			<MyTextLine>And it all breaks down <MyText>at the first</MyText> rehearsal</MyTextLine>
			<MyTextLine>Got the muse in my head she's universal</MyTextLine>
			<MyTextLine>Spinnin' <MyText>me round she's</MyText> coming over me</MyTextLine>
			<MyTextLine>And it all breaks down at the role reversal</MyTextLine>
			<MyTextLine>Got the muse in my head she's universal</MyTextLine>
			<MyTextLine>Spinnin' me round <MyText>she's coming over me</MyText></MyTextLine>
			<MyTextLine>And it all breaks down at the first rehearsal</MyTextLine>
			<MyTextLine>Got <MyText>the muse in</MyText> my head she's universal</MyTextLine>
			<MyTextLine>Spinnin' me round she's coming over me</MyTextLine>
		</div>
	)
}
