#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <clipp.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include "Network.hpp"

void train(const std::string& cfgfile, const std::string& weightfile, const std::vector<int>& gpus)
{
	const char *train_images = "dataset/image_list.txt";
	const char *backup_directory = "backup/";

	srand(time(0));
	char *base = basecfg((char*)cfgfile.c_str());
	printf("%s\n", base);
	float avg_loss = -1;
	network **nets = (network**)calloc(gpus.size(), sizeof(network));

	srand(time(0));
	int seed = rand();
	for (size_t i = 0; i < gpus.size(); ++i) {
		srand(seed);
#ifdef GPU
		cuda_set_device(gpus[i]);
#endif
		nets[i] = load_network((char*)cfgfile.c_str(), (char*)weightfile.c_str(), 0);
		nets[i]->learning_rate *= gpus.size();
	}
	srand(time(0));
	network *net = nets[0];

	int imgs = net->batch * net->subdivisions * gpus.size();
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
	data train, buffer;

	layer l = net->layers[net->n - 1];

	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths((char*)train_images);
	char **paths = (char **)list_to_array(plist);

	load_args args = get_base_args(net);
	args.coords = l.coords;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.jitter = jitter;
	args.num_boxes = l.max_boxes;
	args.d = &buffer;
	args.type = DETECTION_DATA;
	args.threads = 64;

	pthread_t load_thread = load_data(args);
	double time;
	while(get_current_batch(net) < (size_t)net->max_batches){
		time=what_time_is_it_now();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data(args);

		printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

		time=what_time_is_it_now();
		float loss = 0;
#ifdef GPU
		if(gpus.size() == 1){
			loss = train_network(net, train);
		} else {
			loss = train_networks(nets, gpus.size(), train, 4);
		}
#else
		loss = train_network(net, train);
#endif
		if (avg_loss < 0) avg_loss = loss;
		avg_loss = avg_loss*.9 + loss*.1;

		int i = get_current_batch(net);
		printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
		if(i%100==0){
#ifdef GPU
			if(gpus.size() != 1) sync_nets(nets, gpus.size(), 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s.backup", backup_directory, base);
			save_weights(net, buff);
		}
		if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
			if(gpus.size() != 1) sync_nets(nets, gpus.size(), 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
#ifdef GPU
	if(gpus.size() != 1) sync_nets(nets, gpus.size(), 0);
#endif
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}

void test(const std::string& cfgfile, const std::string& weightfile, const std::string& infile, float thresh, const std::string& outfile)
{
	using namespace darknet;

	Network network(cfgfile);
	network.load_weights(weightfile);
	network.set_batch(1);

	Image image = Image(infile);

	auto start = std::chrono::high_resolution_clock::now();

	float* predictions = network.predict(image);

	// const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~";

	const auto& l = network.back();

	size_t dimension_size = l.w * l.h;

	for (int y = 0; y < l.h; ++y) {
		for (int x = 0; x < l.w; ++x) {
			int i = y * l.w + x;
			float objectness = predictions[dimension_size * 4 + i];
			if (objectness < thresh) {
				continue;
			}

			box b;
			b.x = (x + predictions[dimension_size * 0 + i]) / l.w;
			b.y = (y + predictions[dimension_size * 1 + i]) / l.h;
			b.w = exp(predictions[dimension_size * 2 + i]) * l.biases[0] / image.width();
			b.h = exp(predictions[dimension_size * 3 + i]) * l.biases[1] / image.height();

			int left  = (b.x-b.w/2)*image.width();
			int right = (b.x+b.w/2)*image.width();
			int top   = (b.y-b.h/2)*image.height();
			int bot   = (b.y+b.h/2)*image.height();

			image.draw(left, top, right, bot, 0.9f, 0.2f, 0.3f);
		}
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> time = end - start;
	std::cout << time.count() << " seconds" << std::endl;

	image.save(outfile);
}

enum Mode {
	Help,
	Train,
	Test
};

Mode mode;
std::string cfg;
std::string weights;
std::string input;
std::string output = "prediction";
bool nogpu = false;
float thresh = 0.5f;
std::vector<int> gpus = {0};

int main(int argc, char **argv)
{
	using namespace clipp;

	srand(time(0));

	auto cli = (
		command("help").set(mode, Help)
		| ( command("train").set(mode, Train),
#ifdef GPU
			option("--gpus") & values("gpus", gpus),
#endif
			value("cfg", cfg),
			option("weights", weights)
		)
		| (
			command("test").set(mode, Test),
			value("cfg", cfg),
			value("weights", weights),
			value("input", input),
			option("-o", "--output") & value("output", output),
#ifdef GPU
			option("--nogpu").set(nogpu),
#endif
			option("--thresh") & value("thresh", thresh)
		)
	);

	if (!parse(argc, argv, cli)) {
		std::cout << make_man_page(cli, argv[0]) << std::endl;
		return 1;
	}

	switch (mode) {
		case Help:
			std::cout << make_man_page(cli, argv[0]) << std::endl;
			break;
		case Train:
			train(cfg, weights, gpus);
			break;
		case Test:
			test(cfg, weights, input, thresh, output);
			break;
	}

	return 0;
}
