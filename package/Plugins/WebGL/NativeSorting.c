// Native WASM plugin for threaded splat sorting
// This provides background threading support for WebGPU platform

#include <emscripten.h>
#include <emscripten/threading.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>
#include <time.h>

// Maximum number of concurrent sort jobs
#define MAX_SORT_JOBS 256
#define MAX_WORKERS 16

#define true 1
#define false 0

// Sort job structure
typedef struct {
    int* splat_indices;     // External buffer - do not free
    float* positions;       // External buffer - do not free (x,y,z positions, stride of 3)
    int splat_count;
    int position_count;
    float cam_x, cam_y, cam_z;
    int* sorted_indices;    // External buffer - do not free
    atomic_bool is_completed;
    atomic_bool is_active;
    atomic_bool is_assigned; // New field to track if job is assigned to worker
} sort_job_t;

// Worker thread pool structure
typedef struct {
    pthread_t thread;
    atomic_bool is_running;
    atomic_bool should_exit;
} worker_thread_t;

// Global state
static sort_job_t g_sort_jobs[MAX_SORT_JOBS];
static worker_thread_t g_workers[MAX_WORKERS];
static atomic_int g_next_job_id = ATOMIC_VAR_INIT(0);
static atomic_int g_worker_count = ATOMIC_VAR_INIT(4);
static atomic_bool g_is_initialized = ATOMIC_VAR_INIT(false);

// Thread synchronization
static pthread_mutex_t g_job_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_job_available_cond = PTHREAD_COND_INITIALIZER;
static atomic_bool g_shutdown_requested = ATOMIC_VAR_INIT(false);

// Calculate squared distance from camera to splat position
static inline float calc_distance_sq(const float* pos, float cam_x, float cam_y, float cam_z) {
    float dx = pos[0] - cam_x;
    float dy = pos[1] - cam_y;
    float dz = pos[2] - cam_z;
    return dx*dx + dy*dy + dz*dz;
}

// Comparison function for qsort - sort by distance (front to back)
typedef struct {
    float distance;
    int index;
} distance_pair_t;

static int compare_distances(const void* a, const void* b) {
    const distance_pair_t* pair_a = (const distance_pair_t*)a;
    const distance_pair_t* pair_b = (const distance_pair_t*)b;
    
    // Sort front to back (smaller distance first)
    if (pair_a->distance < pair_b->distance) return -1;
    if (pair_a->distance > pair_b->distance) return 1;
    return 0;
}

// Process a single sort job
static void process_sort_job(sort_job_t* job) {
    if (!job || !job->splat_indices || !job->positions || !job->sorted_indices || 
        job->splat_count <= 0 || job->position_count <= 0) {
        if (job) atomic_store(&job->is_completed, true);
        return;
    }
    
    // Allocate distance pairs for sorting
    distance_pair_t* pairs = (distance_pair_t*)malloc(job->splat_count * sizeof(distance_pair_t));
    if (!pairs) {
        atomic_store(&job->is_completed, true);
        return;
    }
    
    // Calculate distances for each splat
    for (int i = 0; i < job->splat_count; i++) {
        int splat_idx = job->splat_indices[i];
        
        // Validate splat index with proper bounds checking
        if (splat_idx >= 0 && splat_idx < job->position_count && 
            (splat_idx * 3 + 2) < (job->position_count * 3)) {
            float* pos = &job->positions[splat_idx * 3];
            // Additional safety check for position pointer
            if (pos != NULL) {
                pairs[i].distance = calc_distance_sq(pos, job->cam_x, job->cam_y, job->cam_z);
            } else {
                pairs[i].distance = 0.0f;
            }
        } else {
            pairs[i].distance = 0.0f; // Invalid indices get distance 0
        }
        pairs[i].index = job->splat_indices[i];
    }
    
    // Sort by distance (front to back)
    qsort(pairs, job->splat_count, sizeof(distance_pair_t), compare_distances);
    
    // Copy sorted indices back
    for (int i = 0; i < job->splat_count; i++) {
        job->sorted_indices[i] = pairs[i].index;
    }
    
    free(pairs);
    atomic_store(&job->is_completed, true);
}

// Find next available job for worker to process
static sort_job_t* find_available_job() {
    for (int i = 0; i < MAX_SORT_JOBS; i++) {
        if (atomic_load(&g_sort_jobs[i].is_active) && 
            !atomic_load(&g_sort_jobs[i].is_assigned) && 
            !atomic_load(&g_sort_jobs[i].is_completed)) {
            // Try to assign this job to current worker
            atomic_bool expected = false;
            if (atomic_compare_exchange_strong(&g_sort_jobs[i].is_assigned, &expected, true)) {
                return &g_sort_jobs[i];
            }
        }
    }
    return NULL;
}

// Worker thread pool function - persistent threads that wait for jobs
static void* worker_thread_pool_func(void* arg) {
    worker_thread_t* worker = (worker_thread_t*)arg;
    atomic_store(&worker->is_running, true);
    
    while (!atomic_load(&worker->should_exit) && !atomic_load(&g_shutdown_requested)) {
        sort_job_t* job = NULL;
        
        // Lock mutex and wait for job
        pthread_mutex_lock(&g_job_queue_mutex);
        
        // Wait for job to become available or shutdown signal
        while (!atomic_load(&g_shutdown_requested) && !atomic_load(&worker->should_exit)) {
            job = find_available_job();
            if (job != NULL) {
                break;
            }
            
            // Wait for job available condition with timeout to check shutdown periodically
            struct timespec timeout;
            #ifdef EMSCRIPTEN
            // Use emscripten's time function for WebGL
            double now = emscripten_get_now() / 1000.0; // Convert ms to seconds
            timeout.tv_sec = (time_t)now;
            timeout.tv_nsec = (long)((now - timeout.tv_sec) * 1000000000.0) + 10000000; // Add 10ms
            #else
            clock_gettime(CLOCK_REALTIME, &timeout);
            timeout.tv_nsec += 10000000; // 10ms timeout
            #endif
            if (timeout.tv_nsec >= 1000000000) {
                timeout.tv_sec += 1;
                timeout.tv_nsec -= 1000000000;
            }
            
            pthread_cond_timedwait(&g_job_available_cond, &g_job_queue_mutex, &timeout);
        }
        
        pthread_mutex_unlock(&g_job_queue_mutex);
        
        // Process the job if found
        if (job != NULL && !atomic_load(&g_shutdown_requested) && !atomic_load(&worker->should_exit)) {
            process_sort_job(job);
            atomic_store(&job->is_assigned, false); // Mark job as no longer assigned
        }
    }
    
    atomic_store(&worker->is_running, false);
    return NULL;
}

// Initialize the native sorting system
EMSCRIPTEN_KEEPALIVE
void NativeSorting_Initialize(int worker_threads) {
    if (atomic_load(&g_is_initialized)) {
        return;
    }
    
    // Clamp worker count
    if (worker_threads <= 0) worker_threads = 4;
    if (worker_threads > MAX_WORKERS) worker_threads = MAX_WORKERS;
    
    #ifdef SINGLE_THREADED
    // Force single threaded mode for fallback builds
    worker_threads = 1;
    emscripten_console_log("NativeSorting: Running in single-threaded fallback mode");
    #endif
    
    atomic_store(&g_worker_count, worker_threads);
    atomic_store(&g_shutdown_requested, false);
    
    // Initialize job array
    memset(g_sort_jobs, 0, sizeof(g_sort_jobs));
    for (int i = 0; i < MAX_SORT_JOBS; i++) {
        atomic_init(&g_sort_jobs[i].is_completed, false);
        atomic_init(&g_sort_jobs[i].is_active, false);
        atomic_init(&g_sort_jobs[i].is_assigned, false);
    }
    
    // Initialize and start worker thread pool
    memset(g_workers, 0, sizeof(g_workers));
    for (int i = 0; i < worker_threads; i++) {
        atomic_init(&g_workers[i].is_running, false);
        atomic_init(&g_workers[i].should_exit, false);
        
        if (pthread_create(&g_workers[i].thread, NULL, worker_thread_pool_func, &g_workers[i]) != 0) {
            emscripten_console_logf("NativeSorting: Failed to create worker thread %d", i);
            // Clean up previously created threads
            for (int j = 0; j < i; j++) {
                atomic_store(&g_workers[j].should_exit, true);
                pthread_join(g_workers[j].thread, NULL);
            }
            // If we can't create any threads, fall back to direct execution mode
            if (i == 0) {
                atomic_store(&g_worker_count, 0);
                emscripten_console_log("NativeSorting: Failed to create worker threads, using direct execution fallback");
            }
            break;
        }
    }
    
    atomic_store(&g_is_initialized, true);
    
    // Debug output
    int actual_workers = atomic_load(&g_worker_count);
    if (actual_workers > 0) {
        emscripten_console_logf("NativeSorting: Initialized with %d worker threads in pool", actual_workers);
    } else {
        emscripten_console_log("NativeSorting: Initialized in direct execution mode (no worker threads)");
    }
}

// Shutdown and cleanup
EMSCRIPTEN_KEEPALIVE
void NativeSorting_Shutdown() {
    if (!atomic_load(&g_is_initialized)) {
        return;
    }
    
    // Signal shutdown to all worker threads
    atomic_store(&g_shutdown_requested, true);
    
    // Wake up all waiting workers
    pthread_mutex_lock(&g_job_queue_mutex);
    pthread_cond_broadcast(&g_job_available_cond);
    pthread_mutex_unlock(&g_job_queue_mutex);
    
    // Wait for all worker threads to exit
    int actual_worker_count = atomic_load(&g_worker_count);
    for (int i = 0; i < actual_worker_count; i++) {
        atomic_store(&g_workers[i].should_exit, true);
        if (atomic_load(&g_workers[i].is_running)) {
            pthread_join(g_workers[i].thread, NULL);
        }
    }
    
    // Clean up active jobs
    for (int i = 0; i < MAX_SORT_JOBS; i++) {
        if (atomic_load(&g_sort_jobs[i].is_active)) {
            // Clear pointers (Unity manages the memory)
            g_sort_jobs[i].splat_indices = NULL;
            g_sort_jobs[i].positions = NULL;
            g_sort_jobs[i].sorted_indices = NULL;
            
            atomic_store(&g_sort_jobs[i].is_active, false);
            atomic_store(&g_sort_jobs[i].is_assigned, false);
            atomic_store(&g_sort_jobs[i].is_completed, false);
        }
    }
    
    atomic_store(&g_is_initialized, false);
    emscripten_console_log("NativeSorting: Worker thread pool shutdown completed");
}

// Get worker thread count
EMSCRIPTEN_KEEPALIVE
int NativeSorting_GetWorkerCount() {
    return atomic_load(&g_worker_count);
}

// Start a sort job with external buffers (no copying)
EMSCRIPTEN_KEEPALIVE
int NativeSorting_StartSortJob(int* splat_indices, int splat_count, 
                               float* positions, int position_count,
                               int* sorted_indices, 
                               float cam_x, float cam_y, float cam_z) {
    if (!atomic_load(&g_is_initialized) || splat_count <= 0 || position_count <= 0) {
        return -1;
    }
    
    // Validate input pointers
    if (!splat_indices || !positions || !sorted_indices) {
        return -1;
    }
    
    // Find available job slot
    int job_id = -1;
    for (int i = 0; i < MAX_SORT_JOBS; i++) {
        if (!atomic_load(&g_sort_jobs[i].is_active)) {
            job_id = i;
            break;
        }
    }
    
    if (job_id == -1) {
        return -1; // No available slots
    }
    
    sort_job_t* job = &g_sort_jobs[job_id];
    
    // Store job data (Unity manages the lifetime of external buffers)
    job->splat_indices = splat_indices;
    job->positions = positions;
    job->sorted_indices = sorted_indices;
    
    job->splat_count = splat_count;
    job->position_count = position_count;
    job->cam_x = cam_x;
    job->cam_y = cam_y;
    job->cam_z = cam_z;
    
    atomic_store(&job->is_completed, false);
    atomic_store(&job->is_assigned, false);
    atomic_store(&job->is_active, true);
    
    // Check if we have worker threads available
    int worker_count = atomic_load(&g_worker_count);
    if (worker_count > 0) {
        // Signal worker threads that a new job is available
        pthread_mutex_lock(&g_job_queue_mutex);
        pthread_cond_signal(&g_job_available_cond);
        pthread_mutex_unlock(&g_job_queue_mutex);
        
        // Debug logging
        //emscripten_console_logf("NativeSorting: Queued job %d with %d splats, %d positions", 
        //                       job_id, splat_count, position_count);
    } else {
        // Direct execution fallback - no worker threads available
        atomic_store(&job->is_assigned, true);
        process_sort_job(job);
        atomic_store(&job->is_assigned, false);
        
        // Debug logging
        //emscripten_console_logf("NativeSorting: Executed job %d directly (no worker threads)", job_id);
    }
    
    return job_id;
}

// Check if job is completed
EMSCRIPTEN_KEEPALIVE
int NativeSorting_IsJobCompleted(int job_id) {
    if (job_id < 0 || job_id >= MAX_SORT_JOBS) {
        return 0;
    }
    
    return atomic_load(&g_sort_jobs[job_id].is_completed) ? 1 : 0;
}

// Note: GetSortedIndices is no longer needed - results are written directly to the output buffer

// Cleanup completed job
EMSCRIPTEN_KEEPALIVE
void NativeSorting_CleanupJob(int job_id) {
    if (job_id < 0 || job_id >= MAX_SORT_JOBS) {
        return;
    }
    
    sort_job_t* job = &g_sort_jobs[job_id];
    
    if (atomic_load(&job->is_active)) {
        // Clear pointers (but don't free - Unity manages the memory)
        job->splat_indices = NULL;
        job->positions = NULL;
        job->sorted_indices = NULL;
        
        atomic_store(&job->is_active, false);
        atomic_store(&job->is_assigned, false);
        atomic_store(&job->is_completed, false);
    }
}
