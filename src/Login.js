// Login.js
import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import Swal from 'sweetalert2';
import { DataContext } from './Data/DataContext';
import { ClipDataContext } from './Data/ClipDataContext';
import './Login.css';

function Login() {
  const [id, setId] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();
  const { setData } = useContext(DataContext);
  const { updateClipInfo } = useContext(ClipDataContext);
  const Server_IP = process.env.REACT_APP_Server_IP;

  const handleLogin = async () => {
    const loginData = {
      ID: id,
      PW: password,
    };

    try {
      const response = await fetch(`${Server_IP}/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(loginData),
      });

      if (response.ok) {
        const data = await response.json();
        const { user_id, user_name, map_camera_info, clip_info, provideo_person_info, camname_info } = data;
        
        // DataContext에 데이터 저장
        setData({
          userId: user_id,
          userName: user_name,
          mapCameraInfo: map_camera_info,
          provideopersoninfo: provideo_person_info,
          camnameinfo: camname_info
        });

        // ClipDataContext에 clipInfo 저장
        updateClipInfo(clip_info);

        // 토큰 저장
        localStorage.setItem('token', data.token);

        Swal.fire({
          icon: 'success',
          title: '로그인 성공',
          text: '로그인 성공을 하셨습니다!',
        });
        navigate('/home');
      } else {
        Swal.fire({
          icon: 'error',
          title: '로그인 실패',
          text: '아이디 또는 비밀번호가 올바르지 않습니다.',
        });
      }
    } catch (error) {
      Swal.fire({
        icon: 'error',
        title: '로그인 오류',
        text: '서버와의 통신 중 오류가 발생했습니다.',
      });
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      handleLogin();
    }
  };

  return (
    <div className="login-form-container ">
      <h2>로그인</h2>
      <input
        type="text"
        placeholder="아이디"
        className="login-input"
        value={id}
        onChange={(e) => setId(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <input
        type="password"
        placeholder="비밀번호"
        className="login-input"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        onKeyDown={handleKeyDown}
      />
      <div className="login-button-container">
        <button className="link-button1" onClick={handleLogin}>
          로그인
        </button>
        <button className="link-button2" onClick={() => navigate('/signup')}>
          회원가입
        </button>
      </div>
    </div>
  );
}

export default Login;
