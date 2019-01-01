
#include "guest_session.hpp"
#include "throw_if_failed.hpp"
#include "string.hpp"
#include "safe_array.hpp"

namespace vbox {
	
GuestSession::GuestSession(IGuestSession* handle): handle(handle) {
	if (!handle) {
		throw std::runtime_error(__PRETTY_FUNCTION__);
	}
}

GuestSession::~GuestSession() {
	if (handle) {
		close();
		IGuestSession_Release(handle);
	}
}

GuestSession::GuestSession(GuestSession&& other): handle(other.handle) {
	other.handle = nullptr;
}

GuestSession& GuestSession::operator=(GuestSession&& other) {
	std::swap(handle, other.handle);
	return *this;
}

GuestSessionWaitResult GuestSession::wait_for(GuestSessionWaitForFlag wait_flag, uint32_t timeout_ms) {
	try {
		GuestSessionWaitResult_T result = GuestSessionWaitResult_None;
		throw_if_failed(IGuestSession_WaitFor(handle, wait_flag, timeout_ms, &result));
		return (GuestSessionWaitResult)result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void GuestSession::close() {
	try {
		throw_if_failed(IGuestSession_Close(handle));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool GuestSession::directory_exists(const fs::path& dir, bool follow_links) {
	try {
		int result = false;
		//IGuestSession_DirectoryExists works well only if directory does exist
		//if it doesn't, it throws an exception (VBOX_E_IPRT_ERROR) instead of returning false
		//and it's the only exception we can expect from this function (see man)
		//So in case the directory doesn't exists we can just ignore the exception
		//and return false. Easy
		IGuestSession_DirectoryExists(handle, StringIn(dir.generic_string()), follow_links, &result);
		
		return (bool)result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void GuestSession::directory_create(const fs::path& dir, uint32_t mode) {
	try {
		SafeArray safe_array;
		throw_if_failed(IGuestSession_DirectoryCreate(handle,
			StringIn(dir.generic_string()),
			mode,
			SAFEARRAY_AS_IN_PARAM(uint32_t, safe_array)));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void GuestSession::directory_remove(const fs::path& dir) {
	try {
		throw_if_failed(IGuestSession_DirectoryRemove(handle, StringIn(dir.generic_string())));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void GuestSession::directory_remove_recursive(const fs::path& dir) {
	try {
		SafeArray safe_array;
		IProgress* result = nullptr;
		throw_if_failed(IGuestSession_DirectoryRemoveRecursive(handle, StringIn(dir.generic_string()), SAFEARRAY_AS_IN_PARAM(uint32_t, safe_array), &result));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

bool GuestSession::file_exists(const fs::path& dir, bool follow_links) {
	try {
		int result = false;
		//See directory_exists comments
		IGuestSession_FileExists(handle, StringIn(dir.generic_string()), follow_links, &result);
		return (bool)result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

void GuestSession::file_remove(const fs::path& path) {
	try {
		throw_if_failed(IGuestSession_FsObjRemove(handle, StringIn(path.generic_string())));
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

Progress GuestSession::file_copy_to_guest(const fs::path& src, const fs::path& destination) {
	try {
		SafeArray safe_array;
		IProgress* result = nullptr;
		throw_if_failed(IGuestSession_FileCopyToGuest(handle,
			StringIn(src.generic_string()),
			StringIn(destination.generic_string()),
			SAFEARRAY_AS_IN_PARAM(uint32_t, safe_array), &result));
		return result;
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

GuestProcess GuestSession::process_create(const fs::path& executable,
		const std::vector<std::string>& arguments,
		const std::vector<std::string>& env_changes,
		const std::vector<ProcessCreateFlag>& flags,
		uint32_t timeout_ms)
{
	try {
		IGuestProcess* result = nullptr;
		std::vector<StringIn> strings_in_args {arguments.begin(), arguments.end()};
		std::vector<BSTR> bstrs_args {strings_in_args.begin(), strings_in_args.end()};

		SafeArray safe_array_args(VT_BSTR, (ULONG)bstrs_args.size());
		safe_array_args.copy_in(bstrs_args.data(), (ULONG)(bstrs_args.size() * sizeof(BSTR)));

		std::vector<StringIn> strings_in_env {env_changes.begin(), env_changes.end()};
		std::vector<BSTR> bstrs_env {strings_in_env.begin(), strings_in_env.end()};

		SafeArray safe_array_env(VT_BSTR, (ULONG)bstrs_env.size());
		safe_array_env.copy_in(bstrs_env.data(), (ULONG)(bstrs_env.size() * sizeof(BSTR)));

		int bitset = 0;
		for (auto flag: flags) {
			bitset |= flag;
		}

		SafeArray safe_array = SafeArray::bitset(bitset);
		throw_if_failed(IGuestSession_ProcessCreate(handle,
			StringIn(executable.generic_string()),
			SAFEARRAY_AS_IN_PARAM(BSTR, safe_array_args),
			SAFEARRAY_AS_IN_PARAM(BSTR, safe_array_env),
			SAFEARRAY_AS_IN_PARAM(ProcessCreateFlag_T, safe_array),
			timeout_ms,
			&result));

		//TODO: new timeout should be general timeout - time for process to be created
		GuestProcess gp(result);
		if (gp.wait_for(ProcessWaitForFlag_Start, timeout_ms) == ProcessWaitResult_Start) {
			return gp;
		} else {
			throw std::runtime_error("Wait for process start error");
		}
	}
	catch (const std::exception&) {
		std::throw_with_nested(std::runtime_error(__PRETTY_FUNCTION__));
	}
}

}
