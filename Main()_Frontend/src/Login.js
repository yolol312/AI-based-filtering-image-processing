// Login.js
import React, { useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import Swal from "sweetalert2";
import { DataContext } from "./Data/DataContext";
import { PersonDataContext } from "./Data/PersonDataContext";
import { RealTimeDataContext } from "./Data/RealTimeDataContext";
import "./Login.css";

function Login() {
  const [id, setId] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();
  const { setData } = useContext(DataContext);
  const { setPersonData } = useContext(PersonDataContext);
  const { setRealTimeData } = useContext(RealTimeDataContext);
  const Server_IP = process.env.REACT_APP_Server_IP;
  const RealTime_Server_IP = process.env.REACT_APP_REALTIME_Server_IP;

  const handleLogin = async () => {
    const loginData = {
      ID: id,
      PW: password,
    };

    const login2Data = {
      user_id: id,
    };
    try {
      // 두 개의 fetch 요청을 병렬로 처리
      const [response1, response2] = await Promise.all([
        fetch(`${Server_IP}/login`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(loginData),
        }),
        fetch(`${RealTime_Server_IP}/realtime_login`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(login2Data),
        }),
      ]);

      if (response1.ok && response2.ok) {
        const data1 = await response1.json();
        const data2 = await response2.json();

        const {
          user_id,
          user_name,
          map_camera_provideo_info,
          camname_info,
          person_info,
        } = data1;

        // DataContext에 데이터 저장
        setData({
          userId: user_id,
          userName: user_name,
          mapcameraprovideoinfo: map_camera_provideo_info,
          camnameinfo: camname_info,
        });

        // PersonDataContext에 person_info 저장
        setPersonData(person_info);

        // 서버 2 응답에서 cam_name을 추출하여 RealTimeDataContext에 저장
        setRealTimeData({ cam_name: data2.cam_name });

        // 토큰 저장
        localStorage.setItem("token", data1.token);

        Swal.fire({
          icon: "success",
          title: "로그인 성공",
          text: "로그인 성공을 하셨습니다!",
        });
        navigate("/home");
      } else {
        Swal.fire({
          icon: "error",
          title: "로그인 실패",
          text: "아이디 또는 비밀번호가 올바르지 않습니다.",
        });
      }
    } catch (error) {
      Swal.fire({
        icon: "error",
        title: "로그인 오류",
        text: "서버와의 통신 중 오류가 발생했습니다.",
      });
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      handleLogin();
    }
  };

  return (
    <div className="login-background">
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
          <button className="link-button2" onClick={() => navigate("/signup")}>
            회원가입
          </button>
        </div>
      </div>
    </div>
  );
}

export default Login;
