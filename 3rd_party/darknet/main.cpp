#include "darknet.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <clipp.h>
#include <iostream>

void train(const std::string& cfgfile, const std::string& weightfile, const std::vector<int>& gpus)
{
	char *train_images = "dataset/image_list.txt";
	char *backup_directory = "backup/";

	srand(time(0));
	char *base = basecfg((char*)cfgfile.c_str());
	printf("%s\n", base);
	float avg_loss = -1;
	network **nets = (network**)calloc(gpus.size(), sizeof(network));

	srand(time(0));
	int seed = rand();
	int i = 0;
	for (auto gpu: gpus) {
		srand(seed);
#ifdef GPU
		cuda_set_device(gpu);
#endif
		nets[i] = load_network((char*)cfgfile.c_str(), (char*)weightfile.c_str(), 0);
		nets[i]->learning_rate *= gpus.size();
		++i;
	}
	srand(time(0));
	network *net = nets[0];

	int imgs = net->batch * net->subdivisions * gpus.size();
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net->learning_rate, net->momentum, net->decay);
	data train, buffer;

	layer l = net->layers[net->n - 1];

	int classes = l.classes;
	float jitter = l.jitter;

	list *plist = get_paths(train_images);
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
	while(get_current_batch(net) < net->max_batches){
		time=what_time_is_it_now();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data(args);

		printf("Loaded: %lf seconds\n", what_time_is_it_now()-time);

		time=what_time_is_it_now();
		float loss = 0;
#ifdef GPU
		if(ngpus == 1){
			loss = train_network(net, train);
		} else {
			loss = train_networks(nets, ngpus, train, 4);
		}
#else
		loss = train_network(net, train);
#endif
		if (avg_loss < 0) avg_loss = loss;
		avg_loss = avg_loss*.9 + loss*.1;

		i = get_current_batch(net);
		printf("%ld: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), what_time_is_it_now()-time, i*imgs);
		if(i%100==0){
#ifdef GPU
			if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s.backup", backup_directory, base);
			save_weights(net, buff);
		}
		if(i%10000==0 || (i < 1000 && i%100 == 0)){
#ifdef GPU
			if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
#ifdef GPU
	if(ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}


void test(const std::string& cfgfile, const std::string& weightfile, const std::string& filename, float thresh, const std::string& outfile)
{
	network *net = load_network((char*)cfgfile.c_str(), (char*)weightfile.c_str(), 0);
	set_batch_network(net, 1);
	srand(2222222);
	double time;
	char buff[256];
	char *input = buff;
	float nms=.45;
	strncpy(input, filename.c_str(), 256);
	image im = load_image_color(input,0,0);
	image sized = letterbox_image(im, net->w, net->h);
	layer l = net->layers[net->n-1];


	float *X = sized.data;
	time=what_time_is_it_now();
	network_predict(net, X);
	printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
	int nboxes = 0;
	detection *dets = get_network_boxes(net, im.w, im.h, thresh, 0.5, 0, 1, &nboxes);
	if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
	draw_detections(im, dets, nboxes, thresh, l.classes);
	free_detections(dets, nboxes);
	save_image(im, outfile.c_str());

	free_image(im);
	free_image(sized);
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
