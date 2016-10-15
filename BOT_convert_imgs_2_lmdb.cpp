#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;
using std::pair;
using boost::scoped_ptr;

std::vector<std::pair<std::string, int> > read_imgs_filelist(
        const std::string& file_path)
{
    std::ifstream infile(file_path.c_str());
    std::vector<std::pair<std::string, int> > imgs;
    std::string line;
    size_t pos;
    int label;

    while(std::getline(infile, line))
    {
        pos = line.find_last_of(' ');
        label = atoi(line.substr(pos + 1).c_str());
        imgs.push_back(std::make_pair(line.substr(0, pos), label));
    }

    infile.close();
    LOG(INFO) << "Finished load image file list";

    return imgs;
}

void convert_imgs_format(std::vector<std::pair<std::string, int> >& imgs,
        const std::string& format, const std::string& db_name, const std::string& root)
{
    const bool is_color = true;
    const bool check_size = false;
    const bool encoded = false;
    const std::string encode_type = "";

    LOG(INFO) << "data total size " << imgs.size();
    LOG(INFO) << "Shuffle data...";
    shuffle(imgs.begin(), imgs.end());

    int resize_height = 224;
    int resize_width = 224;

    scoped_ptr<db::DB> db(db::GetDB(format));
    db->Open(db_name, db::NEW);
    scoped_ptr<db::Transaction> txn(db->NewTransaction());

    std::string root_floder = root;
    //std::string root_floder = "";
    Datum datum;
    int count = 0;
    int data_size = 0;
    bool data_size_initialized = false;

    LOG(INFO) << "Start convert images";
    for (int i = 0; i < imgs.size(); ++ i)
    {
        bool status;
        std::string enc = encode_type;

        if(encoded && !enc.size())
        {}

        status = ReadImageToDatum(root_floder + imgs[i].first,
                imgs[i].second, resize_height, resize_width, is_color,
                enc, &datum);

        if (status == false)
        {
            continue;
        }
        if (check_size)
        {
            if (!data_size_initialized)
            {
                data_size = datum.channels() * datum.height() * datum.width();
                data_size_initialized = true;
            }
            else
            {
                const std::string& data = datum.data();
                CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
                    << data.size();
            }
        }

        std::string key_str = caffe::format_int(i, 8) + "_" + imgs[i].first;

        std::string out;
        CHECK(datum.SerializeToString(&out));
        txn->Put(key_str, out);

        if (++ count % 1000 == 0)
        {
            txn->Commit();
            txn.reset(db->NewTransaction());
            LOG(INFO) << "Processed " << count << " files.";
        }
    }

    if (count % 1000 != 0)
    {
        txn->Commit();
        LOG(INFO) << "Processed " << count << " files.";
    }

    LOG(INFO) << "finished convert images";
}

int main(int argc, char** argv)
{
    const std::string root = "examples/BOT/";

    std::vector<std::pair<std::string, int> > train_imgs = read_imgs_filelist(root + "train.txt");
    std::vector<std::pair<std::string, int> > val_imgs = read_imgs_filelist(root + "val.txt");

    convert_imgs_format(train_imgs, "lmdb", root + "train_imgs_lmdb", root);
    //convert_imgs_format(val_imgs, "lmdb", root + "val_imgs_lmdb", root);
    return 0;
}
