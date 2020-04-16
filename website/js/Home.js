
import React from 'React'
import Layout from './Layout'

module.exports = function() {
	return (
		<Layout>
			<div>
				<div className="index-hero">
					<div className="index-hero-inner">
						<h1 className="index-hero-project-tagline">
							<img alt="Docusaurus with Keytar" className="index-hero-logo" src="/img/docusaurus_keytar.svg"/>Docusaurus makes it easy to maintain <span className="index-hero-project-keywords">Open Source</span> documentation websites.
						</h1>
						<div className="index-ctas">
							<a className="button index-ctas-get-started-button" href="/docs/ru/installation">Начало работы</a>
						</div>
					</div>
				</div>
				<div className="announcement">
					<div className="announcement-inner">If you don't need advanced features such as versioning or translations, try out <a href="https://v2.docusaurus.io">Docusaurus 2</a> instead! Contribute to its roadmap by suggesting features or <a href="https://v2.docusaurus.io/feedback/">giving feedback here</a>!
					</div>
				</div>
				<div className="mainContainer">
					<div className="container lightBackground paddingBottom paddingTop">
						<div className="wrapper">
							<div className="gridBlock">
								<div className="blockElement alignCenter imageAlignTop threeByGridBlock">
									<div className="blockImage">
										<img src="/img/undraw_typewriter.svg" alt="Markdown"/>
									</div>
									<div className="blockContent">
										<h2>
											<div>
												<span>
													<p>Работает на Markdown</p>
												</span>
											</div>
										</h2>
										<div>
											<span>
												<p>Save time and focus on your project's documentation. Simply
														write docs and blog posts with <a href="/docs/ru/doc-markdown">Markdown</a>
														and Docusaurus will publish a set of static html files ready
														to serve.</p>
											</span>
										</div>
									</div>
								</div>
								<div className="blockElement alignCenter imageAlignTop threeByGridBlock">
									<div className="blockImage">
										<img src="/img/undraw_react.svg" alt="React"/>
									</div>
									<div className="blockContent">
										<h2>
											<div>
												<span>
													<p>Создан с использованием React</p>
												</span>
											</div>
										</h2>
										<div>
											<span>
												<p><a href="/docs/ru/api-pages">Extend or customize</a>
												your project's layout by reusing React. Docusaurus can be
												extended while reusing the same header and footer.</p>
											</span>
										</div>
									</div>
								</div>
								<div className="blockElement alignCenter imageAlignTop threeByGridBlock">
									<div className="blockImage">
										<img src="/img/undraw_around_the_world.svg" alt="Translation"/>
									</div>
									<div className="blockContent">
										<h2>
											<div>
												<span>
													<p>Готов для переводов</p>
												</span>
											</div>
										</h2>
										<div>
											<span><p><a href="/docs/ru/translation">Localization</a>
												comes pre-configured. Use <a href="https://crowdin.com/">Crowdin</a> to translate your docs
												into over 70 languages.</p>
											</span>
										</div>
									</div>
								</div>
							</div>
							<br/>
							<br/>
							<div className="gridBlock">
								<div className="blockElement alignCenter imageAlignTop twoByGridBlock">
									<div className="blockImage">
										<img src="/img/undraw_version_control.svg" alt="Document Versioning"/>
									</div>
									<div className="blockContent">
										<h2>
											<div>
												<span>
													<p>Версионирование документов</p>
												</span>
											</div>
										</h2>
										<div>
											<span>
												<p>Support users on all versions of your project. Document
												<a href="/docs/ru/versioning">versioning</a>
												helps you keep documentation in sync with project releases.</p>
											</span>
										</div>
									</div>
								</div>
								<div className="blockElement alignCenter imageAlignTop twoByGridBlock">
									<div className="blockImage">
										<img src="/img/undraw_algolia.svg" alt="Document Search"/>
									</div>
									<div className="blockContent">
										<h2>
											<div>
												<span>
													<p>Поиск по документам</p>
												</span>
											</div>
										</h2>
										<div>
											<span>
												<p>Make it easy for your community to <a href="/docs/ru/search">find</a> what they need in your documentation.
														We proudly support <a href="https://www.algolia.com/">Algolia documentation search</a>.
												</p>
											</span>
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
					<div className="container paddingBottom paddingTop">
						<div className="wrapper">
							<div className="gridBlock">
								<div className="blockElement imageAlignSide imageAlignRight twoByGridBlock">
									<div className="blockContent">
										<h2>
											<div>
												<span>
													<p>Быстрая настройка</p>
												</span>
											</div>
										</h2>
										<div>
											<span>
												<p>Get <a href="/docs/ru/site-creation">up and running</a>
													quickly without having to worry about site design.</p>
											</span>
										</div>
									</div>
									<div className="blockImage">
										<img src="/img/undraw_setup_wizard.svg" alt="Docusaurus on a Scooter"/>
									</div>
								</div>
							</div>
						</div>
					</div>
					<div className="container lightBackground paddingBottom paddingTop">
						<div className="wrapper">
							<div className="gridBlock">
								<div className="blockElement imageAlignSide imageAlignLeft twoByGridBlock">
									<div className="blockImage">
										<img src="/img/docusaurus_live.gif" alt="Docusaurus Demo"/>
									</div>
									<div className="blockContent">
										<h2>
											<div>
												<span>
													<p>Разработка и внедрение</p>
												</span>
											</div>
										</h2>
										<div>
											<span>
												<p>Make design and documentation changes by using the included
													<a href="/docs/ru/site-preparation#verifying-installation">live server</a>.
													<a href="/docs/ru/publishing">Publish</a>
													your site to GitHub pages or other static file hosts
													manually, using a script, or with continuous integration
													like CircleCI.</p>
											</span>
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
					<div className="container paddingBottom paddingTop">
						<div className="wrapper">
							<div className="gridBlock">
								<div className="blockElement imageAlignSide imageAlignRight twoByGridBlock">
									<div className="blockContent">
										<h2>
											<div>
												<span>
													<p>Особенности сайта</p>
												</span>
											</div>
										</h2>
										<div>
											<span>
												<p>Docusaurus currently provides support to help your website
													use <a href="/docs/ru/translation">translations</a>,
													<a href="/docs/ru/search">search</a>,
													and <a href="/docs/ru/versioning">versioning</a>,
													along with some other special <a href="/docs/ru/doc-markdown">documentation markdown features</a>.
													If you have ideas for useful features, feel free to
													contribute on <a href="https://github.com/facebook/docusaurus">GitHub</a>!</p>
											</span>
										</div>
									</div>
									<div className="blockImage">
										<img src="/img/undraw_features_overview.svg" alt="Monochromatic Docusaurus"/>
									</div>
								</div>
							</div>
						</div>
					</div>
					<div className="testimonials">
						<div className="container paddingBottom paddingTop">
							<div className="wrapper">
								<div className="gridBlock">
									<div className="blockElement alignCenter imageAlignTop threeByGridBlock">
										<div className="blockImage">
											<img src="/img/christopher-chedeau.jpg" alt="Christopher &quot;vjeux&quot; Chedeau"/>
										</div>
										<div className="blockContent">
											<h2>
												<div>
													<span>
														<p>Christopher "vjeux" Chedeau <br/>
															<font size="2">Lead Prettier Developer</font>
														</p>
													</span>
												</div>
											</h2>
											<div>
												<span>
													<p>
														<em>I've helped open source many projects at Facebook and every one needed a website. They all had very similar constraints: the documentation should be written in markdown and be deployed via GitHub pages. None of the existing solutions were great, so I hacked my own and then forked it whenever we needed a new website. I’m so glad that Docusaurus now exists so that I don’t have to spend a week each time spinning up a new one.</em>
													</p>
												</span>
											</div>
										</div>
									</div>
									<div className="blockElement alignCenter imageAlignTop threeByGridBlock">
										<div className="blockImage">
											<img src="/img/hector-ramos.png" alt="Hector Ramos"/>
										</div>
										<div className="blockContent">
											<h2>
												<div>
													<span>
														<p>Hector Ramos <br/><font size="2">Lead React Native Advocate</font></p>
													</span>
												</div>
											</h2>
											<div>
												<span>
													<p>
														<em>Open source contributions to the React Native docs have skyrocketed after our move to Docusaurus. The docs are now hosted on a small repo in plain markdown, with none of the clutter that a typical static site generator would require. Thanks Slash!</em>
													</p>
												</span>
											</div>
										</div>
									</div>
									<div className="blockElement alignCenter imageAlignTop threeByGridBlock">
										<div className="blockImage">
											<img src="/img/ricky-vetter.jpg" alt="Ricky Vetter"/>
										</div>
										<div className="blockContent">
											<h2>
												<div>
													<span>
														<p>Ricky Vetter <br/><font size="2">ReasonReact Developer</font></p>
													</span>
												</div>
											</h2>
											<div>
												<span><p><em>Docusaurus has been a great choice for the ReasonML family of projects. It makes our documentation consistent, i18n-friendly, easy to maintain, and friendly for new contributors.</em></p>
												</span>
											</div>
										</div>
									</div>
								</div>
							</div>
						</div>
					</div>
				</div>
			</div>
		</Layout>
	)
}
