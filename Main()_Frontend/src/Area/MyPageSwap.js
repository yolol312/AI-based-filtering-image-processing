// MyPageSwap.js

import React, { useState } from 'react';
import './MyPageSwap.css';

function MyPageSwap() {
  // 가정된 사용자 정보
  const [userInfo, setUserInfo] = useState({
    name: '홍길동',
    userId: 'gildong123',
    password: 'mypassword'
  });

  // 새로운 비밀번호 상태 관리
  const [newPassword, setNewPassword] = useState('');

  // 비밀번호 변경 처리 함수
  const handleChangePassword = () => {
    if (newPassword.trim() === '') {
      alert('새 비밀번호를 입력해주세요.');
      return;
    }
    setUserInfo({ ...userInfo, password: newPassword });
    setNewPassword('');
    alert('비밀번호가 변경되었습니다.');
  };

  return (
    <div className='MyPageSwap'>
      <h2>마이페이지</h2>
      <div className='user-info'>
        <div className='info-item'>
          <label>이름:</label>
          <span>{userInfo.name}</span>
        </div>
        <div className='info-item'>
          <label>아이디:</label>
          <span>{userInfo.userId}</span>
        </div>
        <div className='info-item'>
          <label>비밀번호:</label>
          <span>{userInfo.password}</span>
        </div>
      </div>
      <div className='change-password'>
        <h3>비밀번호 변경</h3>
        <input
          type='password'
          placeholder='새 비밀번호 입력'
          value={newPassword}
          onChange={(e) => setNewPassword(e.target.value)}
        />
        <button onClick={handleChangePassword}>변경</button>
      </div>
    </div>
  );
}

export default MyPageSwap;
