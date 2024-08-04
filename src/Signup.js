// SignUp.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Swal from 'sweetalert2';
import './Signup.css';

function Signup() {
  const [id, setId] = useState('');
  const [pw, setPw] = useState('');
  const [name, setName] = useState('');
  const navigate = useNavigate();
  const Server_IP = process.env.REACT_APP_Server_IP;

  const handleSignupSuccess = async () => {
    if (!id || !pw || !name) {
      Swal.fire({
        icon: 'error',
        title: '회원가입 실패',
        text: '모든 필드를 입력해주세요.',
      });
      return;
    }

    const SignupData = [
      {
        ID: id,
        PW: pw,
        Name: name,
      },
    ];

    console.log('보낼 JSON:', JSON.stringify(SignupData));

    try {
      const response = await fetch(`${Server_IP}/receive_data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(SignupData),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('응답 JSON:', JSON.stringify(data));
        Swal.fire({
          icon: 'success',
          title: '회원가입 성공',
          text: '회원가입 성공을 하셨습니다!',
        });
        navigate('/'); // 로그인 버튼 클릭 시 / 경로로 이동
      } else if (response.status === 409) {
        Swal.fire({
          icon: 'warning',
          title: '회원가입 실패',
          text: '아이디가 이미 존재합니다.',
        });
      } else {
        const data = await response.json();
        Swal.fire({
          icon: 'warning',
          title: '회원가입 실패',
          text: data.error || '회원가입에 실패했습니다.',
        });
      }
    } catch (error) {
      console.error('회원가입 요청 중 오류 발생:', error);
      Swal.fire({
        icon: 'error',
        title: '회원가입 오류',
        text: '서버와의 통신 중 오류가 발생했습니다.',
      });
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter') {
      handleSignupSuccess();
    }
  };

  return (
    <div className="sign-form-container">
      <h2>회원가입</h2>
      <input
        type="text"
        placeholder="아이디"
        className="signup-input"
        value={id}
        onChange={(e) => setId(e.target.value)}
        onKeyDown={handleKeyDown} // 엔터키 이벤트 핸들러 추가
      />
      <input
        type="password"
        placeholder="비밀번호"
        className="signup-input"
        value={pw}
        onChange={(e) => setPw(e.target.value)}
        onKeyDown={handleKeyDown} // 엔터키 이벤트 핸들러 추가
      />
      <input
        type="text"
        placeholder="이름"
        className="signup-input"
        value={name}
        onChange={(e) => setName(e.target.value)}
        onKeyDown={handleKeyDown} // 엔터키 이벤트 핸들러 추가
      />
      <div className="button-container">
        <button className="signup-button" onClick={handleSignupSuccess}>
          회원가입
        </button>
      </div>
    </div>
  );
}

export default Signup;
