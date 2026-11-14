#include <stdio.h>
#include "dertab.h"
int main(){
  /* table order: n ascending, then a ascending, then b ascending */
  int i=0, bad=0;
  for(int n=0;n<=DER_MAXP;n++)
    for(int a=0;a<=n;a++)
      for(int b=0;a+b<=n;b++){
        int c=n-a-b;
        int k=der_key(a,b,c);
        if(k!=i){ if(bad<10) printf("MISMATCH a=%d b=%d c=%d -> der_key=%d, table pos=%d\n",a,b,c,k,i); bad++; }
        i++;
      }
  printf("total keys=%d mismatches=%d DER_OFF entries=%d\n",i,bad,(int)(sizeof(DER_OFF)/sizeof(DER_OFF[0])));
  return 0;
}
