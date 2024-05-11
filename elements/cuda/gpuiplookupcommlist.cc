#include <click/config.h>
#include <click/cuda/cuda_utils.hh>
#include <click/standard/scheduleinfo.hh>
#include <click/args.hh>

#include "gpuiplookupcommlist.hh"

CLICK_DECLS


GPUIPLookup::GPUIPLookup() : _ip_list_len(1), _ip_list_cpu(), _ip_list_gpu() {};

bool GPUIPLookup::cp_ip_route(String s, Route *r_store, bool remove_route, Element *context)
{
    Route r;
    if (!IPPrefixArg(true).parse(cp_shift_spacevec(s), r.addr, r.mask, context))
	return false;
    r.addr &= r.mask;


    String word = cp_shift_spacevec(s);
    if (word == "-")
	/* null gateway; do nothing */;
    else if (IPAddressArg().parse(word, r.gw, context))
	/* do nothing */;
    else
	goto two_words;

    word = cp_shift_spacevec(s);
  two_words:
    if (IntArg().parse(word, r.port) || (!word && remove_route))
	if (!cp_shift_spacevec(s)) { // nothing left
	    *r_store = r;
	    return true;
	}

    return false;
}

void GPUIPLookup::print_route(Route route) {
    printf("IP address: %d.%d.%d.%d\n", (route.addr >> (0 * 8)) & 0xFF, (route.addr >> (1 * 8)) & 0xFF, (route.addr >> (2 * 8)) & 0xFF, (route.addr >> (3 * 8)) & 0xFF);
    printf("Mask: %d.%d.%d.%d\n",       (route.mask >> (0 * 8)) & 0xFF, (route.mask >> (1 * 8)) & 0xFF, (route.mask >> (2 * 8)) & 0xFF, (route.mask >> (3 * 8)) & 0xFF);
    printf("Gateway: %d.%d.%d.%d\n",    (route.gw >> (0 * 8)) & 0xFF,   (route.gw >> (1 * 8)) & 0xFF,   (route.gw >> (2 * 8)) & 0xFF,   (route.gw >> (3 * 8)) & 0xFF);
    printf("Port: %d\n", route.port);
}

int GPUIPLookup::configure(Vector<String> &conf, ErrorHandler *errh) {

    int ret = 0, r1, eexist = 0;
    struct Route route;

    _verbose = true;

    _ip_list_cpu = (Route*) malloc(sizeof(Route) * conf.size());

    if (Args(conf, this, errh)
        .bind(conf)
        .read_p("CAPACITY", _capacity)
        .consume() < 0)
    {
        return -1;
    }
    
    // printf("capacity: %d\n", _capacity);


    for (int i = 0; i < conf.size(); i++) {
        if (!cp_ip_route(conf[i], &route, false, this)) {
            errh->error("argument %d should be %<ADDR/MASK [GATEWAY] OUTPUT%>", i+1);
            ret = -EINVAL;
        } else if (route.port < 0 || route.port >= noutputs()) {
            errh->error("argument %d bad OUTPUT", i+1);
            ret = -EINVAL;
        }

        // print_route(route);

        memcpy(&(_ip_list_cpu[i]), &route, sizeof(Route));
        _ip_list_len = conf.size();
    }
    if (eexist)
	errh->warning("%d %s replaced by later versions", eexist, eexist > 1 ? "routes" : "route");
    return ret;
}

int GPUIPLookup::initialize(ErrorHandler *errh) {
    _usable_threads = get_pushing_threads();
    
    cudaMalloc(&_ip_list_gpu, _ip_list_len*sizeof(RouteGPU));
    cudaMemcpy(_ip_list_gpu, _ip_list_cpu, _ip_list_len*sizeof(RouteGPU), cudaMemcpyHostToDevice);


    for (int th_id = 0; th_id < master()->nthreads(); th_id++) {
        if (!_usable_threads[th_id])
            continue;

        state &s = _state.get_value(th_id);

        s.task = new Task(this);
        ScheduleInfo::initialize_task(this, s.task, false, errh);
        s.task->move_thread(th_id);

        s.timer = new Timer(this);
        s.timer->initialize(this);
        s.timer->move_thread(th_id);
        s.timer->schedule_now();

        s.next_list_get = 0;
        s.next_list_put = 0;
        s.comm_lists = new comm_list_state[_lists_per_core];
        for (uint8_t list_id = 0; list_id < _lists_per_core; list_id++) {
            s.comm_lists[list_id].comm_list_put_index = 0;
            s.comm_lists[list_id].comm_list_get_index = 0;
            s.comm_lists[list_id].comm_list_size = _capacity;        
            s.comm_lists[list_id].comm_list = rte_gpu_comm_create_list(0, s.comm_lists[list_id].comm_list_size);   // todo generify GPU
            if (s.comm_lists[list_id].comm_list == NULL)  {
                return errh->error("Unable to create the GPU communication list for thread %d", th_id);
            }

            cudaError_t cuda_ret = cudaStreamCreate(&s.comm_lists[list_id].cuda_stream);
            CUDA_CHECK(cuda_ret, errh);
            wrapper_ip_lookup_persistent(s.comm_lists[list_id].comm_list, s.comm_lists[list_id].comm_list_size, _blocks_per_q, _max_batch, s.comm_lists[list_id].cuda_stream, _ip_list_gpu, _ip_list_len);
            CUDA_CHECK(cudaGetLastError(), errh);
        }
    }

    if (_verbose) {
        errh->message("%s{element}: %d usable threads, %d queue per thread, %d batch per queue.",
            this,
            _usable_threads.weight(), 1, _capacity
        );
    }

    return 0;
}

bool GPUIPLookup::run_task(Task *task) {
    enum rte_gpu_comm_list_status status;
    uint32_t n;
    struct rte_mbuf **pkts;

    // read lists in round robin fashion
    struct comm_list_state *list_state = &_state->comm_lists[_state->next_list_get];
    _state->next_list_get = (_state->next_list_get + 1) % _lists_per_core;

    rte_gpu_comm_get_status(&list_state->comm_list[list_state->comm_list_get_index], &status);
    if (status != RTE_GPU_COMM_LIST_DONE) {
        task->fast_reschedule();
        return false;
    }

    rte_rmb();

    /* Batch the processed packets */
    PacketBatch *head = nullptr;
    WritablePacket *packet, *last;
    n = list_state->comm_list[list_state->comm_list_get_index].num_pkts;
    pkts = list_state->comm_list[list_state->comm_list_get_index].mbufs;
    for (uint32_t i = 0; i < n; i++) {
        packet = static_cast<WritablePacket *>(Packet::make(pkts[i]));
        if (head == NULL) 
            head = PacketBatch::start_head(packet);
        else
            last->set_next(packet);
        last = packet;
    }
    if (head) {
        head->make_tail(last, n);
        output_push_batch(0, head);
    }

    rte_gpu_comm_cleanup_list(&list_state->comm_list[list_state->comm_list_get_index]);
    rte_mb();

    list_state->comm_list_get_index = (list_state->comm_list_get_index + 1) % list_state->comm_list_size;
    return true;
}

void GPUIPLookup::cleanup(CleanupStage cs) {
    cleanup_base(cs);
    free(_ip_list_cpu);
    cudaFree(_ip_list_gpu);
}

ELEMENT_REQUIRES(batch cuda)
CLICK_ENDDECLS
EXPORT_ELEMENT(GPUIPLookup)