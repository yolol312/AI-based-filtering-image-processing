import React, { useState, useContext, useEffect, useRef } from "react";
import "./FileUploadFiltering.css";
import Swal from "sweetalert2";
import { DataContext } from "../Data/DataContext";
import { ProgressBarDataContext } from "../Data/ProgressBarDataContext";

const Server_IP = process.env.REACT_APP_Server_IP;

function FileUploadFiltering({ handleClose }) {
  const { data, setData } = useContext(DataContext); // DataContext에서 setData 가져오기
  const { updateProgressBarInfo, ProgressBarInfo, setProgressBarInfo } =
    useContext(ProgressBarDataContext);

  const [year, setYear] = useState("2024");
  const [month, setMonth] = useState("08");
  const [day, setDay] = useState("13");
  const [hour, setHour] = useState("15");
  const [minute, setMinute] = useState("30");
  const [second, setSecond] = useState("00");
  const [startTime, setStartTime] = useState("");
  const [videoFile, setVideoFile] = useState(null);
  const [address, setAddress] = useState("");
  const [camName, setCamName] = useState("");
  const [bundleName, setbundleName] = useState(""); // 구역 이름 상태 추가
  const [list, setList] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [age, setAge] = useState("");
  const [upclothes, setupClothes] = useState("");
  const [downclothes, setdownClothes] = useState("");
  const [gender, setGender] = useState("");
  const [createClip, setCreateClip] = useState(true);
  const [addressOptions, setAddressOptions] = useState([]);
  const [filteredCamOptions, setFilteredCamOptions] = useState([]);
  const [isDisabled, setIsDisabled] = useState(false); // 버튼 비활성화 상태 추가
  const [selectedIndex, setSelectedIndex] = useState(null); // 추가: 선택된 리스트 항목 인덱스
  const completeRef = useRef(null);
  const intervalRef = useRef(null);
  const [width, setWidth] = useState(0); // width 상태 추가

  const userId = data.userId;

  useEffect(() => {
    if (data.mapcameraprovideoinfo && data.mapcameraprovideoinfo.length > 0) {
      const uniqueAddresses = [
        ...new Set(data.mapcameraprovideoinfo.map((cam) => cam.address)),
      ];
      setAddressOptions(uniqueAddresses);
    }
  }, [data.mapcameraprovideoinfo]);

  useEffect(() => {
    setStartTime(formatDateTime(year, month, day, hour, minute, second));
  }, [year, month, day, hour, minute, second]);

  const formatDateTime = (year, month, day, hour, minute, second) => {
    return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
  };

  useEffect(() => {
    if (address) {
      const cams = data.mapcameraprovideoinfo.filter(
        (cam) => cam.address === address
      );
      setFilteredCamOptions(cams);
      // setCamName(""); // 주소 변경 시 캠 이름 초기화 (제거)
    } else {
      setFilteredCamOptions([]);
    }
  }, [address, data.mapcameraprovideoinfo]);

  const handleAddToList = async () => {
    if (!videoFile || !address || !camName) {
      //|| !startTime) {
      Swal.fire({
        icon: "error",
        title: "오류",
        text: "모든 필드를 입력하세요.",
      });
      return;
    }

    if (videoFile.type !== "video/mp4") {
      Swal.fire({
        icon: "error",
        title: "오류",
        text: "오직 MP4 파일만 업로드 가능합니다.",
      });
      return;
    }

    try {
      const videoContent = await toBase64(videoFile);

      const requestData = {
        user_id: userId,
        video_data: [
          {
            video_name: videoFile.name,
            video_content: videoContent,
          },
        ],
      };

      const response = await fetch(`${Server_IP}/upload_progresstime`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestData),
      });

      if (response.ok) {
        const responseData = await response.json();
        const { total_length } = responseData;

        // total_length 값을 ProgressBarDataContext에 업데이트
        updateProgressBarInfo(total_length);

        Swal.fire({
          icon: "success",
          title: "성공",
          text: `비디오가 성공적으로 업로드되었습니다.`, //총 길이: ${responseData.total_length}초
        });

        setList([
          ...list,
          {
            file: videoFile,
            fileName: videoFile.name,
            address,
            camName,
            startTime,
          },
        ]);
        setVideoFile(null);
        setAddress("");
        setCamName("");
        setStartTime("");
        document.getElementById("videoFileInput").value = "";
      } else {
        console.error("서버 응답 에러:", response.statusText);
        const responseData = await response.json();
        Swal.fire({
          icon: "error",
          title: "실패",
          text: responseData.error || "비디오 업로드에 실패했습니다.", // 서버에서 반환된 오류 메시지를 사용
        });
      }
    } catch (error) {
      console.error("비디오 업로드 중 오류 발생:", error);
      Swal.fire({
        icon: "error",
        title: "오류",
        text: "비디오 업로드 중 오류가 발생했습니다.",
      });
    }
  };

  const handleRemoveFromList = () => {
    if (selectedIndex !== null && selectedIndex < list.length) {
      const newList = list.filter((_, idx) => idx !== selectedIndex);

      setList(newList);

      setSelectedIndex(null);
    } else {
      Swal.fire({
        icon: "error",
        title: "오류",
        text: "삭제할 항목을 선택해주세요.",
      });
    }
  };

  const handleClearList = () => {
    if (list.length > 0) {
      Swal.fire({
        title: "리스트 초기화",
        text: "리스트를 초기화하시겠습니까?",
        icon: "warning",
        showCancelButton: true,
        confirmButtonText: "예",
        cancelButtonText: "아니요",
      }).then((result) => {
        if (result.isConfirmed) {
          setList([]); // 리스트 초기화
          setSelectedIndex(null);
          Swal.fire("초기화 완료", "리스트가 초기화되었습니다.", "success");
        }
      });
    } else {
      Swal.fire("오류", "초기화할 리스트가 없습니다.", "error");
    }
  };

  const resetFilters = () => {
    setbundleName("");
    setSelectedFile(null); // 이미지 파일 초기화
    setAge("");
    setupClothes("");
    setdownClothes("");
    setGender("");
  };

  const handleFileChange = (event) => {
    setVideoFile(event.target.files[0]);
  };

  const handleImageFileChange = (event) => {
    const file = event.target.files[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedFile(file);
    } else {
      Swal.fire({
        icon: "error",
        title: "오류",
        text: "이미지 파일만 업로드 가능합니다.",
      });
      event.target.value = ""; // 파일 입력 필드 초기화
    }
  };

  const startProgress = (duration) => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    if (completeRef.current) {
      clearInterval(completeRef.current);
    }

    setWidth(0); // 초기화 후 일정 시간 대기
    const intervalTime = (duration * 1000) / 100; // 100%를 달성하기 위한 시간 간격 계산

    setTimeout(() => {
      intervalRef.current = setInterval(() => {
        setWidth((prevWidth) => {
          if (prevWidth >= 97) {
            clearInterval(intervalRef.current);
            return 97;
          }
          return prevWidth + 1;
        });
      }, intervalTime); // 동적으로 계산된 시간 간격
    }, 100); // 100ms 대기 후 시작
  };

  const completeProgress = () => {
    if (completeRef.current) {
      clearInterval(completeRef.current);
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    completeRef.current = setInterval(() => {
      setWidth((prevWidth) => {
        if (prevWidth >= 100) {
          clearInterval(completeRef.current);
          setTimeout(() => {
            setWidth(0); // 성공 후 프로그레스 바 초기화
          }, 500); // 0.5초 대기 후 초기화
          return 100;
        }
        return prevWidth + 1;
      });
    }, 20); // 20ms 간격으로 애니메이션 효과
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      if (completeRef.current) {
        clearInterval(completeRef.current);
      }
    };
  }, []);

  const handleSubmit = async () => {
    if (list.length === 0) {
      Swal.fire({
        icon: "error",
        title: "오류",
        text: "최소 하나의 비디오 파일을 리스트에 추가하세요.",
      });
      return;
    }

    if (!age && !upclothes && !downclothes && !gender && !selectedFile) {
      Swal.fire({
        icon: "error",
        title: "오류",
        text: "최소 하나의 필터 조건을 설정하세요.",
      });
      return;
    }

    try {
      setIsDisabled(true); // 모든 입력 필드와 버튼 비활성화

      const videoData = await Promise.all(
        list.map(async (item) => ({
          video_name: item.fileName,
          video_content: await toBase64(item.file),
          address: item.address,
          cam_name: item.camName,
          start_time: item.startTime,
        }))
      );

      let imageData = {};
      if (selectedFile) {
        imageData = {
          image_name: selectedFile.name,
          image_content: await toBase64(selectedFile),
        };
      }

      const user_data = {
        user_id: userId,
      };

      const bundle_data = {
        bundle_name: bundleName,
      };

      const filter_data = {
        age: age || null,
        uptype: upclothes || null,
        downtype: downclothes || null,
        gender: gender || null,
      };

      const flag_data = {
        clip_flag: createClip ? "true" : "false",
      };

      const data = {
        video_data: videoData,
        user_data: user_data,
        filter_data: filter_data,
        flag_data: flag_data,
        bundle_data: bundle_data,
      };

      if (selectedFile) {
        data.image_data = imageData;
      }

      console.log("보낼 JSON:", JSON.stringify(data));
      console.log("bundleName:", bundleName); // 추가된 로그

      startProgress(ProgressBarInfo);

      const response = await fetch(`${Server_IP}/upload_file`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });

      // 완료 버튼 기능
      const responseData = await response.json(); // 응답 데이터 파싱
      if (response.ok) {
        setData((prevData) => ({
          ...prevData,
          filterid: responseData.filter_id,
        })); // filter_id를 DataContext에 저장
        Swal.fire({
          icon: "success",
          title: "성공",
          text: "파일이 성공적으로 업로드되었습니다.",
        }).then(() => {
          handleClose();
          resetFilters(); // 필터 초기화
        });
        completeProgress();
        setIsDisabled(false); // 성공 시 다시 활성화
      } else {
        Swal.fire({
          icon: "error",
          title: "실패",
          text: "파일 업로드에 실패했습니다.",
        });
        setIsDisabled(false); // 업로드 실패 시 다시 활성화
      }
    } catch (error) {
      console.error("파일 업로드 중 오류 발생:", error);
      Swal.fire({
        icon: "error",
        title: "오류",
        text: "파일 업로드 중 오류가 발생했습니다.",
      });
      setIsDisabled(false); // 업로드 실패 시 다시 활성화
    }
  };

  const toBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(",")[1]);
      reader.onerror = (error) => reject(error);
    });
  };

  return (
    <div className="File-modal-body">
      <div className="file-upload-filtering">
        <div className="file-upload">
          <h2>지도 | 카메라 선택</h2>
          <select
            value={address}
            onChange={(e) => setAddress(e.target.value)}
            disabled={isDisabled} // 비활성화 상태에 따라 비활성화
          >
            <option value="">지도 선택</option>
            {[...new Set(addressOptions)].map((addr, index) => (
              <option key={index} value={addr}>
                {addr}
              </option>
            ))}
          </select>
          <select
            value={camName}
            onChange={(e) => setCamName(e.target.value)}
            disabled={!address || isDisabled} // 주소 없거나 비활성화 상태에 따라 비활성화
          >
            <option value="">카메라 선택</option>
            {[...new Set(filteredCamOptions.map((cam) => cam.cam_name))].map(
              (camName, index) => (
                <option key={index} value={camName}>
                  {camName}
                </option>
              )
            )}
          </select>
          <h2>영상 업로드</h2>
          <input
            type="file"
            id="videoFileInput"
            accept="video/mp4"
            onChange={handleFileChange}
            disabled={isDisabled} // 비활성화 상태에 따라 비활성화
          />
          <div className="buttons">
            <button
              onClick={handleAddToList}
              className="add-to-list-button"
              disabled={isDisabled} // 비활성화 상태에 따라 비활성화
            >
              리스트 추가
            </button>
            <button
              onClick={handleRemoveFromList}
              className="remove-from-list-button"
              disabled={isDisabled || selectedIndex === null}
            >
              리스트 삭제
            </button>
          </div>

          <ul className="file-list">
            {list.map((item, index) => (
              <li
                key={index}
                onClick={() => setSelectedIndex(index)}
                className={selectedIndex === index ? "selected" : ""}
              >
                <span className="file-name">{item.fileName}</span>
                <span className="address">{item.address}</span>
                <span className="cam-name">{item.camName}</span>
              </li>
            ))}
          </ul>

          <div className="buttons">
            <button
              onClick={handleClearList}
              className="clear-list-button"
              disabled={isDisabled || list.length === 0}
            >
              리스트 초기화
            </button>
          </div>
        </div>

        <div className="filtering">
          <div className="form-group">
            <h2>필터 이름 지정</h2>
            <input
              type="text"
              value={bundleName}
              onChange={(e) => setbundleName(e.target.value)}
              placeholder="묶음 이름 입력"
              disabled={isDisabled} // 비활성화 상태에 따라 비활성화
            />
            <h2>필터링 선택</h2>
            <div className="image-preview-container">
              <div className="image-preview">
                {selectedFile ? (
                  <img src={URL.createObjectURL(selectedFile)} alt="preview" />
                ) : (
                  "얼굴 이미지"
                )}
              </div>
              <div className="file-input-container">
                <label>얼굴 필터</label>
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageFileChange}
                  disabled={isDisabled} // 비활성화 상태에 따라 비활성화
                />
              </div>
            </div>
          </div>
          <div className="form-group">
            {/* 나이, 상의, 하의, 성별 섹션 */}
            <div className="inline-group">
              <h2>나이</h2>
              <select
                value={age}
                onChange={(e) => setAge(e.target.value)}
                disabled={isDisabled} // 비활성화 상태에 따라 비활성화
              >
                <option value="">선택하지 않음</option>
                <option value="Child">유아</option>
                <option value="Youth">청년</option>
                <option value="Middle">중년</option>
                <option value="Old">노년</option>
              </select>
            </div>
          </div>
          <div className="form-group">
            <div className="inline-group">
              <h2>상의</h2>
              <select
                value={upclothes}
                onChange={(e) => setupClothes(e.target.value)}
                disabled={isDisabled} // 비활성화 상태에 따라 비활성화
              >
                <option value="">선택하지 않음</option>
                <option value="longsleevetop">긴팔상의</option>
                <option value="shortsleevetop">반팔상의</option>
                <option value="sleeveless">민소매</option>
              </select>
            </div>
          </div>
          <div className="form-group">
            <div className="inline-group">
              <h2>하의</h2>
              <select
                value={downclothes}
                onChange={(e) => setdownClothes(e.target.value)}
                disabled={isDisabled} // 비활성화 상태에 따라 비활성화
              >
                <option value="">선택하지 않음</option>
                <option value="shorts">반바지</option>
                <option value="pants">긴바지</option>
                <option value="skirt">치마</option>
              </select>
            </div>
          </div>
          <div className="form-group">
            <div className="inline-group">
              <h2>성별</h2>
              <div className="radio-group">
                <label>
                  <input
                    type="radio"
                    value="남성"
                    checked={gender === "남성"}
                    onChange={(e) => setGender(e.target.value)}
                    disabled={isDisabled} // 비활성화 상태에 따라 비활성화
                  />
                  남성
                </label>
                <label>
                  <input
                    type="radio"
                    value="여성"
                    checked={gender === "여성"}
                    onChange={(e) => setGender(e.target.value)}
                    disabled={isDisabled} // 비활성화 상태에 따라 비활성화
                  />
                  여성
                </label>
              </div>
            </div>
          </div>
          <div className="File-modal-footer">
            <button
              className="File-modal-button-close"
              onClick={handleSubmit}
              disabled={isDisabled} // 비활성화 상태에 따라 비활성화
            >
              저장
            </button>
          </div>
        </div>
      </div>
      <div id="progress-container">
        <div id="progress-bar" style={{ width: `${width}%` }}>
          {width}%
        </div>
      </div>
    </div>
  );
}

export default FileUploadFiltering;
