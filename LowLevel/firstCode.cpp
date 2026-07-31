#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>


int main(){

    // open connection to i2c
    int fd= open("/dev/i2c-1",O_RDWR);
    
    if (fd<0){
        perror("Failed to open i2c bus");
        return 0;
    }

    // select slave
    int ctl = ioctl(fd,I2C_SLAVE,0x33);
    
    if(ctl <0){
        perror("Failed to aquire bus access");
    }

    // 


}

int ReadConstants(){

    uint8_t reg[2];
    uint8_t data[2];

    //read KVDD
    reg[0] = 0x2B;
    reg[1] = 0x24 & 0b00000111;
    if(write(fd,reg,2)!=2){
        perror("Address write failed kvdd");
    }
    if (read(fd, data,2)!=2){
        perror("kvdd read failed");
    }
    uint16_t kvdd = data[0] | (data[1]<< 8);
    if ( kvdd > 1023){
        kvdd -= 2048;
    }
    kvdd = kvdd * (2**5)

    //read KT
    reg[0] = 0x2A;
    reg[1] = 0x24 & 0b00000111;
    if(write(fd,reg,2)!=2){
        perror("Address write failed kt");
    }
    if (read(fd, data,2)!=2){
        perror("kt read failed");
    }
    uint16_t kt = data[0] | (data[1]<< 8);


}

int ConvertOneTemperature(uint16_t PTAT){



}