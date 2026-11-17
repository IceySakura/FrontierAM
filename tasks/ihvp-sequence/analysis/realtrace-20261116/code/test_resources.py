from unittest.mock import patch,MagicMock
import psutil
from resources import same_process,stop_recorded

def test_identity_and_control():
 p=MagicMock();p.create_time.return_value=123.;p.children.return_value=[]
 with patch('resources.psutil.Process',return_value=p):
  assert same_process(20,123)
  assert not same_process(20,122)
  assert not stop_recorded({'pid':20,'birth':122});p.terminate.assert_not_called()
  assert stop_recorded({'pid':20,'birth':123});p.terminate.assert_called_once()
 with patch('resources.psutil.Process',side_effect=psutil.NoSuchProcess(20)):
  assert not same_process(20,123)
 print('PID reuse, stale records, process departure and authorized stop checks PASS')
if __name__=='__main__':test_identity_and_control()
