import React, { useState, useContext, useEffect } from "react";
import "./MyPageSwap.css";
import { DataContext } from "../Data/DataContext";
import { RealTimeDataContext } from "../Data/RealTimeDataContext";
import Swal from "sweetalert2";

function MyPageSwap() {
  const { data } = useContext(DataContext);
  const { RealTimeData, setRealTimeData } = useContext(RealTimeDataContext);
  const Server_IP = process.env.REACT_APP_Server_IP;
  const RealTime_Server_IP = process.env.REACT_APP_REALTIME_Server_IP;

  // 새로운 비밀번호와 현재 비밀번호 상태 관리
  const [newPassword, setNewPassword] = useState("");
  const [currentPassword, setCurrentPassword] = useState("");

  // 캠 이름 상태 관리
  const [newCamName, setNewCamName] = useState("");

  // 초기 상태를 빈 객체로 설정
  const [userInfo, setUserInfo] = useState({});

  // data가 로드되면 userInfo를 업데이트
  useEffect(() => {
    if (data) {
      setUserInfo({
        name: data.userName, // userName에서 이름 가져오기
        userId: data.userId, // userId에서 아이디 가져오기
      });
    }
  }, [data]);

  // 비밀번호 변경 처리 함수
  const handleChangePassword = async () => {
    if (currentPassword.trim() === "" || newPassword.trim() === "") {
      Swal.fire({
        title: "오류",
        text: "현재 비밀번호와 새 비밀번호를 모두 입력해주세요.",
        icon: "error",
        confirmButtonText: "확인",
      });
      return;
    }

    const requestData = {
      user_data: {
        user_id: data.userId,
        user_pw: currentPassword, // 현재 비밀번호
        new_pw: newPassword, // 새 비밀번호
      },
    };

    try {
      const response = await fetch(`${Server_IP}/password_update`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      const data = await response.json();

      if (response.ok) {
        Swal.fire({
          title: "비밀번호가 변경되었습니다.",
          icon: "success",
          confirmButtonText: "확인",
        });
        setCurrentPassword("");
        setNewPassword("");
      } else {
        // 서버에서 발생한 오류 처리
        Swal.fire({
          title: "오류",
          text: data.error || "비밀번호 변경 중 오류가 발생했습니다.",
          icon: "error",
          confirmButtonText: "확인",
        });
      }
    } catch (err) {
      console.error("비밀번호 변경 중 오류 발생: ", err);
      Swal.fire({
        title: "오류",
        text: "서버와의 통신 중 오류가 발생했습니다.",
        icon: "error",
        confirmButtonText: "확인",
      });
    }
  };

  // 캠 이름 변경 처리 함수
  const handleChangeCamName = async () => {
    if (newCamName.trim() === "") {
      alert("새 캠이름을 입력해주세요.");
      return;
    }

    Swal.fire({
      title: "캠 이름을 변경하시겠습니까?",
      text: "변경 시 이전 모든 정보가 삭제됩니다.",
      icon: "warning",
      showCancelButton: true,
      confirmButtonText: "삭제",
      cancelButtonText: "취소",
    }).then(async (result) => {
      if (result.isConfirmed) {
        const requestData = [
          {
            user_id: data.userId,
            cam_name: newCamName, // 새 캠 이름
          },
        ];

        try {
          const response = await fetch(
            `${RealTime_Server_IP}/realtime_cam_update`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify(requestData),
            }
          );

          const data = await response.json();

          // 서버 응답에 따라 메시지를 처리합니다.
          if (response.ok) {
            if (data.status === "success") {
              console.log("캠 이름 변경이 성공적으로 처리되었습니다:", data);

              // 서버 응답에서 cam_name을 RealTimeDataContext에 저장
              setRealTimeData({ cam_name: data.new_cam_name });

              setNewCamName(""); // 캠 이름 변경 성공 시 초기화
              Swal.fire({
                title: "캠 이름이 변경되었습니다.",
                icon: "success",
                confirmButtonText: "확인",
              });

              setNewCamName(""); // 캠 이름 변경 성공 시 초기화
              Swal.fire({
                title: "캠 이름이 변경되었습니다.",
                icon: "success",
                confirmButtonText: "확인",
              });

              setNewCamName(""); // 캠 이름 변경 성공 시 초기화
              Swal.fire({
                title: "캠 이름이 변경되었습니다.",
                icon: "success",
                confirmButtonText: "확인",
              });
            } else {
              console.error("오류 발생: ", data.message);
              Swal.fire({
                title: "오류",
                text: data.message || "캠 이름 변경 중 문제가 발생했습니다.",
                icon: "error",
                confirmButtonText: "확인",
              });
            }
          } else {
            console.error("오류 발생: ", data.message);
            Swal.fire({
              title: "오류",
              text: data.message || "캠 이름 변경 중 문제가 발생했습니다.",
              icon: "error",
              confirmButtonText: "확인",
            });
          }
        } catch (err) {
          console.error("캠 이름 변경 중 오류 발생: ", err);
          Swal.fire({
            title: "캠 이름 변경 중 오류가 발생했습니다.",
            icon: "error",
            confirmButtonText: "확인",
          });
        }
      }
    });
  };

  return (
    <div className="MyPageSwap">
      <h2>마이페이지</h2>
      <div className="user-info">
        <div className="info-item">
          <label>이름:</label>
          <span>{userInfo.name}</span>
        </div>
        <div className="info-item">
          <label>아이디:</label>
          <span>{userInfo.userId}</span>
        </div>
      </div>
      <div className="change-container">
        <div className="change-password">
          <h3>비밀번호 변경</h3>
          <div className="password-inputs">
            <input
              type="password"
              placeholder="현재 비밀번호 입력"
              value={currentPassword}
              onChange={(e) => setCurrentPassword(e.target.value)}
            />
            <input
              type="password"
              placeholder="새 비밀번호 입력"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
            />
          </div>
          <button className="change-button" onClick={handleChangePassword}>
            변경
          </button>
        </div>
        <div className="change-password">
          <h3>실시간 캠 이름 변경</h3>
          <div className="password-inputs">
            <input
              type="text"
              placeholder="새 캠이름 입력"
              value={newCamName}
              onChange={(e) => setNewCamName(e.target.value)}
            />
          </div>
          <button className="change-button" onClick={handleChangeCamName}>
            변경
          </button>
          <label className="labelstyle">
            캠 이름 변경은 캠 위치 변경을 뜻합니다.{" "}
          </label>
          <label className="labelstyle">
            따라서 이전 위치의 캠 정보들은 삭제됩니다.{" "}
          </label>
        </div>
      </div>
    </div>
  );
}

export default MyPageSwap;
