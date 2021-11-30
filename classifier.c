#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>      
#include <sys/types.h>  
#include <sys/wait.h>  
#include <string.h>
#include <math.h>
#include "knn.h"

/**
 * main() takes in the following command line arguments.
 *   -K <num>:  K value for kNN (default is 1)
 *   -d <distance metric>: a string for the distance function to use
 *          euclidean or cosine (or initial substring such as "eucl", or "cos")
 *   -p <num_procs>: The number of processes to use to test images
 *   -v : If this argument is provided, then print additional debugging information
 *        (You are welcome to add print statements that only print with the verbose
 *         option.  We will not be running tests with -v )
 *   training_data: A binary file containing training image / label data
 *   testing_data: A binary file containing testing image / label data
 *   (Note that the first three "option" arguments (-K <num>, -d <distance metric>,
 *   and -p <num_procs>) may appear in any order, but the two dataset files must
 *   be the last two arguments.
 * 
 * You need to do the following:
 *   - Parse the command line arguments, call `load_dataset()` appropriately.
 *   - Create the pipes to communicate to and from children
 *   - Fork and create children, close ends of pipes as needed
 *   - All child processes should call `child_handler()`, and exit after.
 *   - Parent distributes the test dataset among children by writing:
 *        (1) start_idx: The index of the image the child should start at
 *        (2)    N:      Number of images to process (starting at start_idx)
 *     Each child should get at most N = ceil(test_set_size / num_procs) images
 *      (The last child might get fewer if the numbers don't divide perfectly)
 *   - Parent waits for children to exit, reads results through pipes and keeps
 *      the total sum.
 *   - Print out (only) one integer to stdout representing the number of test 
 *      images that were correctly classified by all children.
 *   - Free all the data allocated and exit.
 *   - Handle all relevant errors, exiting as appropriate and printing error message to stderr
 */
void usage(char *name) {
    fprintf(stderr, "Usage: %s -v -K <num> -d <distance metric> -p <num_procs> training_list testing_list\n", name);
}

int main(int argc, char *argv[]) {

    int opt;
    int K = 1;             // default value for K
    char *dist_metric = "euclidean"; // default distant metric
    int num_procs = 1;     // default number of children to create
    int verbose = 0;       // if verbose is 1, print extra debugging statements
    int total_correct = 0; // Number of correct predictions

    while((opt = getopt(argc, argv, "vK:d:p:")) != -1) {
        switch(opt) {
        case 'v':
            verbose = 1;
            break;
        case 'K':
            K = atoi(optarg);
            break;
        case 'd':
            dist_metric = optarg;
            break;
        case 'p':
            num_procs = atoi(optarg);
            break;
        default:
            usage(argv[0]);
            exit(1);
        }
    }

    if(optind >= argc) {
        fprintf(stderr, "Expecting training images file and test images file\n");
        exit(1);
    } 

    char *training_file = argv[optind];
    optind++;
    char *testing_file = argv[optind];
  
    // Set which distance function to use
    /* You can use the following string comparison which will allow
     * prefix substrings to match:
     * 
     * If the condition below is true then the string matches
     * if (strncmp(dist_metric, "euclidean", strlen(dist_metric)) == 0){
     *      //found a match
     * }
     */ 
  
    // TODO
    double (*metric)(Image *a, Image *b);
    if (strncmp(dist_metric, "euclidean", strlen(dist_metric)) == 0) {
        metric = distance_euclidean;
    } else if (strncmp(dist_metric, "cosine", strlen(dist_metric)) == 0) {
        metric = distance_cosine;
    } else {
        fprintf(stderr, "Expected any initial substring of \"euclidean\" or \"cosine\" as argument for -d\n");
        exit(1);        
    }


    // Load data sets
    if(verbose) {
        fprintf(stderr,"- Loading datasets...\n");
    }
    
    Dataset *training = load_dataset(training_file);
    if ( training == NULL ) {
        fprintf(stderr, "The data set in %s could not be loaded\n", training_file);
        exit(1);
    }

    Dataset *testing = load_dataset(testing_file);
    if ( testing == NULL ) {
        fprintf(stderr, "The data set in %s could not be loaded\n", testing_file);
        exit(1);
    }

    // Create the pipes and child processes who will then call child_handler.
    // Distribute the work to the children by writing their starting index and
    // the number of test images to process to their write pipe
    if(verbose) {
        printf("- Creating children ...\n");
    }

    // TODO
    int from_children[num_procs * 2];

    int start_idx = 0;
    int boundary = testing->num_items % num_procs;
    int N;

    for (int i = 0; i < num_procs; i++) {

        if (i < boundary) {
            N = ceil( (double)testing->num_items / num_procs);
        } else {
            N = floor( (double)testing->num_items / num_procs);
        }

        int *c_to_p = from_children + 2*i;
        // Pipe to send data from child to parent
        if (pipe(c_to_p) == -1) {
            perror("pipe");
            exit(1);
        }
        // Pipe to send data from parent to child
        int p_to_c[2];
        if (pipe(p_to_c) == -1) {
            perror("pipe");
            exit(1);
        }

        if (write(p_to_c[1], &start_idx, sizeof(int)) == -1) {
            perror("write");
            exit(1);
        }
        if (write(p_to_c[1], &N, sizeof(int)) == -1) {
            perror("write");
            exit(1);
        }

        if (close(p_to_c[1]) < 0) {
            perror("close");
            exit(1);
        }
        
        // Make child and manage
        int k = fork();
        if (k == 0) { // Child process

            // Close read end of c_to_p
            if (close(c_to_p[0]) < 0) {
                perror("close");
                exit(1);
            }

            child_handler(training, testing, K, metric, p_to_c[0], c_to_p[1]);

            // Close all unnecessary pipe ends

            if (close(p_to_c[0]) < 0) {
                perror("close");
                exit(1);
            }
            if (close(c_to_p[1]) < 0) {
                perror("close");
                exit(1);
            }

            // Free datasets since their instance is also created for each child
            free_dataset(training);
            free_dataset(testing);

            // Child should stop here
            exit(0);

        } else if (k < 0) { // Some error
            perror("fork");
            exit(1);
        }
        // Back to parent

        // Close all pipe ends except read end of c_to_p (to be used later)
        if (close(p_to_c[0]) < 0) {
            perror("close");
            exit(1);
        }
        if (close(c_to_p[1]) < 0) {
            perror("close");
            exit(1);
        }

        // Update start_idx for next iteration
        start_idx += N;
    }

    // Read results from children through their pipe
    // TODO
    for (int i = 0; i < num_procs; i++) {
        int fd = from_children[2 * i];
        int count;

        int bool = 1;
        // Keep reading from pipe till an integer appears
        while (bool) {
            int num_read = read(fd, &count, sizeof(int));
            if (num_read == sizeof(int)) {
                bool = 0;
                total_correct += count;
            } else if (num_read == -1) {
                perror("read");
                exit(1);
            }
        }

        if (close(fd) < 0) {
            perror("close");
            exit(1);
        }
    }

    // Wait for children to finish
    if(verbose) {
        printf("- Waiting for children...\n");
    }

    // TODO
    for (int i = 0; i < num_procs; i++) {
        int status;
        if (wait(&status) < 0) {
            perror("wait");
            exit(1);
        }
        if (WIFEXITED(status)) {
            if (WEXITSTATUS(status) == 1) {
                fprintf(stderr, "Problem with reading or writing in children processes");
                exit(1);
            }
        }
        
    }




    if(verbose) {
        printf("Number of correct predictions: %d\n", total_correct);
    }

    // This is the only print statement that can occur outside the verbose check
    printf("%d\n", total_correct);

    // Clean up any memory, open files, or open pipes

    // TODO
    free_dataset(training);
    free_dataset(testing);

    return 0;
}
