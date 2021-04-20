#include "server.h"
#include <sys/socket.h> 
#include <sys/ioctl.h>
#include <netinet/in.h> 

#define BUFFER_LEN 8*1024*1024
#define PORT 4747

int server_sock, client_sock; 
struct sockaddr_in address; 
int opt = 1; 
int addrlen = sizeof(address); 
char buffer[BUFFER_LEN] = {0}; 
uint32_t command = 1;
uint32_t datalen = 0;

void start_server() {
	// Creating socket file descriptor 
    if ((server_sock = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
       
    // Forcefully attaching socket to the port 8080 
    if (setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, 
                                                  &opt, sizeof(opt))) 
    { 
        perror("setsockopt"); 
        exit(EXIT_FAILURE); 
    } 

    //Get IP address of eth0
    struct ifreq ifr;
    ifr.ifr_addr.sa_family = AF_INET;
    strncpy(ifr.ifr_name, "eth0", IFNAMSIZ-1);
    ioctl(server_sock, SIOCGIFADDR, &ifr);

    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = ((struct sockaddr_in *)&ifr.ifr_addr)->sin_addr.s_addr; 
    address.sin_port = htons( PORT ); 
       
    // Forcefully attaching socket to the port 8080 
    if (bind(server_sock, (struct sockaddr *)&address,  
                                 sizeof(address))<0) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 
    if (listen(server_sock, 1) < 0) 
    { 
        perror("listen"); 
        exit(EXIT_FAILURE); 
    } 
    printf("Wainting for connection at %s\n", address.sin_addr.s_addr)
    if ((client_sock = accept(server_sock))<0) 
    { 
        perror("accept"); 
        exit(EXIT_FAILURE); 
    } 
    printf("Client connected\n");
}

void capture_trigger()
{
	send(client_sock, (char*)&command, sizeof(command), 0);
    read(client_sock, (char*)&datalen, 4);
    printf("Packet size: %i\n", datalen);
    int ptr = 0;
    while(ptr < datalen) {
    	ptr += read(client_sock, &(buffer[ptr]), BUFFER_LEN);
    }
}
void stop_server() {
	shutdown(client_sock, 2);
	shutdown(server_sock, 2);
}