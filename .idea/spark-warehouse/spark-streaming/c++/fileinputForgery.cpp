    #include <iostream>
    #include <stdio.h>
    #include <unistd.h>
    #include <dirent.h>
    #include <stdlib.h>
    #include <sys/stat.h>
    #include <string.h>
    #include <vector>

	#include<sys/mman.h>
	#include<unistd.h>
	#include<fcntl.h>

    using namespace std;

    /***** Global Variables *****/
    char dir[100] = "/home";
    int const MAX_STR_LEN = 200;

    /* Show all files under dir_name , do not show directories ! */
    vector<char *> showAllFiles( const char * dir_name )
    {
        // check the parameter !
        if( NULL == dir_name )
        {
            cout<<" dir_name is null ! "<<endl;
        }

        // check if dir_name is a valid dir
        struct stat s;
        lstat( dir_name , &s );
        if( ! S_ISDIR( s.st_mode ) )
        {
            cout<<"dir_name is not a valid directory !"<<endl;

        }

        struct dirent * filename;    // return value for readdir()
        DIR * dir;                   // return value for opendir()
        dir = opendir( dir_name );
        if( NULL == dir )
        {
            cout<<"Can not open dir "<<dir_name<<endl;

        }
        cout<<"Successfully opened the dir !"<<endl;

        vector<char *> results;
        /* read all the files in the dir ~ */
        while( ( filename = readdir(dir) ) != NULL )
        {
            // get rid of "." and ".."
            if( strcmp( filename->d_name , "." ) == 0 ||
                strcmp( filename->d_name , "..") == 0    )
                continue;
//            cout<<filename ->d_name <<endl;
            results.push_back(filename ->d_name);
        }
        return  results;
    }

    int file_size2(char* filename)
    {
        struct stat statbuf;
        stat(filename,&statbuf);
        int size=statbuf.st_size;
        return size;
    }

    int leng(const char* des){
       int len=0;
       while(true){
    //	  printf("ok \n");
    	  if((char)*(des+len)!='\0')
    		len=len+1;
    	  else
    		break;
       }
       return len;
    }


    void fileinput(string filename,const char*name){
    	char *p;
    	int len = filename.length();
    	p=(char *)malloc((len+1)*sizeof(char));
    	filename.copy(p,len,0);

    	int fd;
    	char *mapped_mem;
    	int flength = 1024;
    	void * start_addr = 0;
    	fd = open(p, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
    	flength = file_size2(p);
        //write(fd, '\0', 1); /* 在文件最后添加一个空字符，以便下面printf正常工作 */
    	//printf("s %d",flength);
    	mapped_mem = (char *)mmap(start_addr, flength, PROT_READ, //允许读
    	MAP_PRIVATE, //不允许其它进程访问此内存区域
    	fd, 0);
    	char* list=new char[100];
    	list[0]='#';
    	int i=0;
    	while(true)
    	 {list[i]=mapped_mem[i];
    	  printf("wo shi,%c \n",list[i]);
    	  if(list[i]=='\n')
    	    break;
    	  i++;
    	 }
    	list[i]='\0';
    	printf("%s \n",list);
    	close(fd);
    	munmap(mapped_mem, flength);
    }

    string m_replace(string str,string pattern,string dstPattern,int count=-1)
    {
        string retStr="";
        string::size_type pos;
        int i=0,l_count=0,szStr=str.length();
        if(-1 == count) // replace all
            count = szStr;
        for(i=0; i<szStr; i++)
        {
            if(string::npos == (pos=str.find(pattern,i)))  break;
            if(pos < szStr)
            {
                retStr += str.substr(i,pos-i) + dstPattern;
                i=pos+pattern.length()-1;
                if(++l_count >= count)
                {
                    i++;
                    break;
                }
            }
        }
        retStr += str.substr(i);
        return retStr;
    }

    int main()
    {   // 测试
    	const char* dir="/lf/2017.11.14/test_data/";
    	char* dir_N="/lf/2017.11.14/test_data/";
//    	printf("%d",std::string::max_size());
    	vector<char *> results=showAllFiles(dir);
    	for(int i=0; i<results.size(); i++)
    	   {char* total=new char[100];
    		strcat(total,dir_N);
    	    strcat(total,results[i]);
            cout<<"this is"<<total<<endl;
            string str=total;
            string file=results[i];
            string file_new=m_replace(file,".txt","");
            printf("that is %s \n",file_new.c_str());
            fileinput(str,file_new.c_str());
            printf("--------------------------------");
    	   }
        return 0;
    }
