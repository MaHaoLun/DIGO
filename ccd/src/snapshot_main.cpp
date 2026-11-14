#include "sdc/io.hpp"

#include <iostream>
#include <map>
#include <string>

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: sdc_snapshot --workspace-root ROOT --input-dir data/staged/PACKAGE --city BJ|XA --date 20160101|20160102 --run-id NEW\n";
            return 0;
        }
        std::map<std::string, std::string> args;
        for (int i = 1; i < argc; i += 2) {
            const std::string key(argv[i]);
            if (key != "--workspace-root" && key != "--input-dir" && key != "--city" && key != "--date" && key != "--run-id") throw sdc::io::InputError("unknown option: " + key);
            if (i + 1 >= argc) throw sdc::io::InputError("missing value for " + key);
            if (!args.emplace(key, argv[i+1]).second) throw sdc::io::InputError("duplicate option: " + key);
        }
        for (const char* key : {"--workspace-root", "--input-dir", "--city", "--date", "--run-id"}) if (!args.count(key)) throw sdc::io::InputError(std::string("required option: ") + key);
        return sdc::io::build_snapshot({args.at("--workspace-root"), args.at("--input-dir"), args.at("--city"), args.at("--date"), args.at("--run-id")});
    } catch (const sdc::io::BoundaryViolation& error) {
        std::cerr << "BOUNDARY_VIOLATION " << error.what() << '\n'; return 3;
    } catch (const sdc::io::InputError& error) {
        std::cerr << "INPUT_ERROR " << error.what() << '\n'; return 2;
    } catch (const std::invalid_argument& error) {
        std::cerr << "INPUT_ERROR " << error.what() << '\n'; return 2;
    } catch (const std::exception& error) {
        std::cerr << "SNAPSHOT_ERROR " << error.what() << '\n'; return 1;
    }
}
