// SignUp.js
import React, { useState, useContext } from "react";
import { useNavigate } from "react-router-dom";
import { RealTimeDataContext } from "./Data/RealTimeDataContext";
import Swal from "sweetalert2";
import "./Signup.css";

function Signup() {
  const { setRealTimeData } = useContext(RealTimeDataContext);
  const [id, setId] = useState("");
  const [pw, setPw] = useState("");
  const [name, setName] = useState("");
  const [camname, setCamName] = useState("");
  const navigate = useNavigate();
  const Server_IP = process.env.REACT_APP_Server_IP;
  const RealTime_Server_IP = process.env.REACT_APP_REALTIME_Server_IP;

  const handleSignupSuccess = async () => {
    if (!id || !pw || !name || !camname) {
      Swal.fire({
        icon: "error",
        title: "회원가입 실패",
        text: "모든 필드를 입력해주세요.",
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

    const Signup2Data = [
      {
        user_id: id,
        cam_name: camname,
      },
    ];

    console.log("서버 1에 보낼 JSON:", JSON.stringify(SignupData));
    console.log("서버 2에 보낼 JSON:", JSON.stringify(Signup2Data));

    try {
      // 두 개의 fetch 요청을 병렬로 처리
      const [response1, response2] = await Promise.all([
        fetch(`${Server_IP}/receive_data`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(SignupData),
        }),
        fetch(`${RealTime_Server_IP}/realtime_signup`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(Signup2Data),
        }),
      ]);

      // 두 요청 모두 성공했는지 확인
      if (response1.ok && response2.ok) {
        const data1 = await response1.json();
        const data2 = await response2.json();
        console.log("응답 JSON (서버 1):", JSON.stringify(data1));
        console.log("응답 JSON (서버 2):", JSON.stringify(data2));

        Swal.fire({
          icon: "success",
          title: "회원가입 성공",
          text: "회원가입 성공을 하셨습니다!",
        });
        navigate("/"); // 로그인 버튼 클릭 시 / 경로로 이동
      } else if (response1.status === 409 || response2.status === 409) {
        Swal.fire({
          icon: "warning",
          title: "회원가입 실패",
          text: "아이디가 이미 존재합니다.",
        });
      } else {
        const data1 = await response1.json();
        const data2 = await response2.json();
        Swal.fire({
          icon: "warning",
          title: "회원가입 실패",
          text: data1.error || data2.error || "회원가입에 실패했습니다.",
        });
      }
    } catch (error) {
      console.error("회원가입 요청 중 오류 발생:", error);
      Swal.fire({
        icon: "error",
        title: "회원가입 오류",
        text: "서버와의 통신 중 오류가 발생했습니다.",
      });
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      handleSignupSuccess();
    }
  };

  return (
    <div className="signup-background">
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
        <input
          type="text"
          placeholder="실시간 캠 이름 지정"
          className="signup-input"
          value={camname}
          onChange={(e) => setCamName(e.target.value)}
          onKeyDown={handleKeyDown} // 엔터키 이벤트 핸들러 추가
        />
        <div className="button-container">
          <button className="signup-button" onClick={handleSignupSuccess}>
            회원가입
          </button>
        </div>
      </div>
    </div>
  );
}

export default Signup;
